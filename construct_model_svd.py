from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_feature as train, validate as validate
from models.util import FcLayer,ReLayer1,SepFcLayer,BottleLayer,SepAtt,MobileFcLayer
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from models.resnetv14 import Transform


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')

    parser.add_argument('-r', '--gamma', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=9.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=100.0, help='weight balance for other losses')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40,50', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])

    # dataset
    parser.add_argument('--model', type=str, default='ResNet18',
                        choices=['ResNet18', 'ResNet50', 'resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44',
                                 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', ])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=4, help='the experiment id')

    parser.add_argument('--backbone_path', type=str,
                        default='./save/models/selfdis/ResNet34_cifar100_lr_0.05_decay_0.0005_trial_10/ResNet34_best.pth',
                        help='teacher model snapshot')
    parser.add_argument('--fc_path', type=str,
                        default='./save/models/selfdis/ResNet34_cifar100_lr_0.05_decay_0.0005_trial_10/fc_best.pth',
                        help='teacher model snapshot')
    parser.add_argument('--tran_path2', type=str,
                        default='./save/models/selfdis/ResNet50_cifar100_lr_0.05_decay_0.0005_trial_1/transform46_best.pth',
                        help='teacher model snapshot')
    parser.add_argument('--tran_path1', type=str,
                        default='./save/models/selfdis/ResNet50_cifar100_lr_0.05_decay_0.0005_trial_1/transform14_best.pth',
                        help='teacher model snapshot')

    parser.add_argument('-channel', help='Last layer of second model', type=int, default=64)
    parser.add_argument('--init', choices=['random', 'perm', 'eye', 'ps_inv', 'ones-zeros'], default='ps_inv')

    opt = parser.parse_args()
    opt.teachertest_path = './result'

    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models/feature'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    vallog_headers = [
        'model',
        'trial',
        'bestepoch',
        'bestacc',
    ]
    if not os.path.exists(opt.teachertest_path):
        os.makedirs(opt.teachertest_path)
    if not os.path.exists(os.path.join(opt.teachertest_path, 'feature_test.csv')):
        with open(os.path.join(opt.teachertest_path, 'feature_test.csv'), 'w') as f:
            f.write(','.join(vallog_headers) + '\n')

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model_t = model_t
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'],strict=False)
    print('==> done')
    return model

def load_Spe(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model_t = model_t
    modelSpe = model_dict[model_t](num_classes=n_cls)
    #modelSpe.load_state_dict(torch.load(model_path)['model'],strict=False)
    print('==> done')
    return modelSpe

def load_reduction1(model_path):
    print('==> loading fc model')
    relayer1 = ReLayer1()
    relayer1.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return relayer1

def load_tran1(model_path):
    print('==> loading fc model')
    transform14 = Transform(512, 512)
    print("transform14",transform14)
    transform14.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return transform14

def load_tran2(model_path):
    print('==> loading fc model')
    transform46 = Transform(1024, 1024)
    print("transform46",transform46)
    transform46.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return transform46

def init_front():
    print('==> loading fc model')
    modelfront = load_Spe(opt.backbone_path, 100)
    model = load_teacher(opt.backbone_path, 100)
    relayer1 = load_reduction1(opt.fc_path)
    modelfront.conv1 = model.conv1
    modelfront.bn1 = model.bn1
    modelfront.layer1 = model.layer1
    modelfront.layer2 = model.layer2
    modelfront.layer3 = model.layer3
    modelfront.scala1 = model.layer3
    modelfront.linear = relayer1.fc7

    print("modelSpe",modelfront)
    #transform46.load_state_dict(torch.load(model_path)['model'],strict=False)
    return modelfront

def init_front():
    print('==> loading fc model')
    modelfront = load_Spe(opt.backbone_path, 100)
    modelfront.conv1 = model.conv1
    modelfront.bn1 = model.bn1
    modelfront.layer1 = model.layer1
    modelfront.layer2 = relayer1.scala1
    modelfront.layer3 = relayer1.scala4
    modelfront.layer4 = relayer1.scala6
    modelfront.linear = relayer1.fc3

    print("modelSpe",modelfront)
    #transform46.load_state_dict(torch.load(model_path)['model'],strict=False)
    return modelfront



def main():
    best_acc = 0
    best_epoch = 0

    opt = parse_option()
    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        train_loader, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)

        n_cls = 10
    elif opt.dataset == 'tiny-imagenet':
        train_loader, val_loader = get_imagenet_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 200
    else:
        raise NotImplementedError(opt.dataset)




    # model
    modelSpe = load_Spe(opt.backbone_path, 100)
    print("modelSpe",modelSpe)

    model = load_teacher(opt.backbone_path, 100)
    print("model", model)


    relayer1 = load_reduction1(opt.fc_path)
    print("relayer1",relayer1)
    transform14 = load_tran1(opt.tran_path1)
    transform46=load_tran2(opt.tran_path2)

    for name in transform46.state_dict():
        print(name)


    rank = 10

    def rsvd(input, rank):
        """
        Randomized SVD torch function
        Extremely fast computation of the truncated Singular Value Decomposition, using
        randomized algorithms as described in Halko et al. 'finding structure with randomness
        usage :
        Parameters:
        -----------
        * input : Tensor (2D matrix) whose SVD we want
        * rank : (int) number of components to keep
        Returns:
        * (u,s,v) : tuple, classical output as the builtin torch svd function
        """
        assert len(input.shape) == 2, "input tensor must be 2D"
        (m, n) = input.shape
        p = torch.min(torch.tensor([2 * rank, n]))
        x = torch.randn(n, p, device=input.device)
        y = torch.matmul(input, x)

        # get an orthonormal basis for y
        uy, sy, _ = torch.svd(y)
        rcond = torch.finfo(input.dtype).eps * m
        tol = sy.max() * rcond
        num = torch.sum(sy > tol)
        W1 = uy[:, :num]

        B = torch.matmul(W1.T, input)
        W2, s, v = torch.svd(B)
        u = torch.matmul(W1, W2)
        k = torch.min(torch.tensor([rank, u.shape[1]]))
        return (u[:, :k], s[:k], v[:, :k])

    # a=transform46.state_dict()['transform.0.weight'].data
    # # u,s,v=torch.linalg.svd(a)
    # print("a",a.shape)
    # start = time.time()
    # (ufull, sfull, vfull) = torch.svd(a)
    # print('   torch.svd: %0.1fms' % (1000 * (time.time() - start)))
    # start = time.time()
    # (u, s, v) = rsvd(a, rank=rank)
    # print('   rsvd (%d components): %0.1fms' % (rank, 1000 * (time.time() - start)))
    # print('errors:')
    # reconstructed_full = torch.matmul(
    #     ufull[:, :rank], torch.matmul(torch.diag(sfull[:rank]), vfull[:, :rank].T))
    # reconstructed_rsvd = torch.matmul(u, torch.matmul(torch.diag(s), v.T))
    # print('   fast vs truncated full: %f' % torch.norm(
    #     reconstructed_full - reconstructed_rsvd))
    # print('   input vs fast: %f' % torch.norm(
    #     a - reconstructed_rsvd))


    conv1_weight = transform46.state_dict()['transform.0.weight'].data
    print("conv1_weight",conv1_weight.shape)
    x_flat = conv1_weight.view(-1, conv1_weight.shape[1])
    print("x_flat", x_flat.shape)
    u, s, v = torch.svd(x_flat)
    print("u", u)

    k = 128
    u_k = u[:, :k]
    s_k = s[:k]
    v_k = v[:, :k]

    # 重构张量
    x_svd = torch.matmul(torch.matmul(u_k, torch.diag(s_k)), v_k.t())
    x_svd = x_svd.view(conv1_weight.shape)
    print("x_svd", x_svd.shape)
    print(torch.norm(conv1_weight - x_svd))


    # net.state_dict()['conv1.weight'] = torch.from_numpy(conv1_weight)

    # net.state_dict()['conv2.weight'] = torch.from_numpy(conv2_weight)
    # print(net.state_dict()['conv2.weight'].shape)

    torch.save(transform46.state_dict(), 'old.pth')
    new_state_dict = torch.load('old.pth')
    new_state_dict['transform.0.weight'] = x_svd.data
    torch.save(new_state_dict, 'svd.pth')
    transform46.load_state_dict(torch.load('svd.pth'))
    transform46.eval()

    conv2_weight = transform14.state_dict()['transform.0.weight'].data
    print("conv1_weight", conv2_weight.shape)
    x_flat = conv2_weight.view(-1, conv2_weight.shape[1])
    print("x_flat", x_flat.shape)
    u, s, v = torch.svd(x_flat)
    print("u", u)

    k = 64
    u_k = u[:, :k]
    s_k = s[:k]
    v_k = v[:, :k]

    # 重构张量
    x_svd = torch.matmul(torch.matmul(u_k, torch.diag(s_k)), v_k.t())
    x_svd = x_svd.view(conv2_weight.shape)

    # net.state_dict()['conv1.weight'] = torch.from_numpy(conv1_weight)

    # net.state_dict()['conv2.weight'] = torch.from_numpy(conv2_weight)
    # print(net.state_dict()['conv2.weight'].shape)

    torch.save(transform14.state_dict(), 'old14.pth')
    new_state_dict = torch.load('old14.pth')
    new_state_dict['transform.0.weight'] = x_svd.data
    torch.save(new_state_dict, 'svd14.pth')
    #transform14.load_state_dict(torch.load('svd14.pth'))
    transform14.eval()

    # 定义 SVD 压缩函数
    def svd_compress(weight, k):
        # 将权重矩阵转换为2D矩阵
        weight_2d = weight.view(weight.size(0) * weight.size(1), -1)
        U, S, V = torch.svd(weight_2d)
        print("S",S)
        # 保留前k个奇异值
        U_k = U[:, :k]
        S_k = S[:k]
        V_k = V.t()[:k, :]
        # 重构图像
        compressed_weight = torch.mm(torch.mm(U_k, torch.diag(S_k)), V_k)
        return compressed_weight.view(weight.size())

    # 对模型中的每个卷积层进行 SVD 压缩
    for name, module in relayer1.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Compressing {name}...")
            new_weight = svd_compress(module.weight.data, k=1)  # 选择保留的奇异值数量
            module.weight.data = new_weight.view_as(module.weight)

    compressed_model_path = 'path_to_your_compressed_resnet8.pth'
    torch.save(relayer1.state_dict(), compressed_model_path)

    relayer1.load_state_dict(torch.load('path_to_your_compressed_resnet8.pth'))
    relayer1.eval()


    modelSpe.conv1 = model.conv1
    modelSpe.bn1 = model.bn1

    modelSpe.layer1 = model.layer1
    modelSpe.layer2 = model.layer2
    modelSpe.layer3 = model.layer3
    modelSpe.layer4 = relayer1.scala6

    modelSpe.linear=relayer1.fc3



    state = {
        'model': modelSpe.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
    torch.save(state, save_file)
    print('saving the best model!')

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        modelSpe = modelSpe.cuda()
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    test_acc, test_acc_top5, test_loss = validate(val_loader, modelSpe, criterion, opt)

    print('best accuracy:', test_acc)

    with open(os.path.join(opt.teachertest_path, 'teacher_test.csv'), 'a') as f:
        log = [opt.model, opt.trial, best_epoch, test_acc.item()]
        log = map(str, log)
        f.write(','.join(log) + '\n')



if __name__ == '__main__':
    main()
