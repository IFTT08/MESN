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
from dataset.cifar10 import get_cifar10_dataloaders
from dataset.tiny_imagenet import get_imagenet_dataloader

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_self as train, test_self as test
from models.util import FcLayer,ReLayer2,BottleLayer,Conv1FcLayer,ReLayer1,SepAtt
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss,CosSimilarity
from models.MetaEmbedding import MetaEmbedding
from adaptive_inference import dynamic_evaluate
from thop import profile

def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    parser.add_argument('-r', '--gamma', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=10.0, help='weight balance for other losses')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=3, help='temperature for KD distillation')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])

    # dataset
    parser.add_argument('--model', type=str, default='ResNet18',
                        choices=['ResNet18', 'ResNet34','ResNet50','ResNet101', 'resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44',
                                 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', ])
    parser.add_argument('--dataset', type=str, default='tiny-imagenet', choices=['cifar100','cifar10','tiny-imagenet'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')
    parser.add_argument('--save', default='save/default-{}'.format(time.time()),
                           type=str, metavar='SAVE',
                           help='path to the experiment logging directory'
                                '(default: save/debug)')
    parser.add_argument('--nBlocks', type=int, default=4)
    parser.add_argument('--path_backbone', type=str, default='./save/models/selfdis/ResNet18_tiny-imagenet_lr_0.05_decay_0.0005_trial_23/ResNet18_best.pth',
                        help='teacher model snapshot')
    parser.add_argument('--path_fc', type=str,
                        default='./save/models/selfdis/ResNet18_tiny-imagenet_lr_0.05_decay_0.0005_trial_23/fc_best.pth',
                        help='teacher model snapshot')
    parser.add_argument('--path_emb', type=str,
                        default='./save/models/selfdis/MobileNetV2_cifar10_lr_0.01_decay_0.0005_trial_1/embedding_best.pth',
                        help='teacher model snapshot')

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
        opt.model_path = './save/models/selfdis'
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
        'testloss',
        'bestacc',
        'bestacc1_4',
        'bestacc2_4',
        'bestacc3_4',
        'bestacc4_4',
        'bestaccens',
    ]
    if not os.path.exists(opt.teachertest_path):
        os.makedirs(opt.teachertest_path)
    if not os.path.exists(os.path.join(opt.teachertest_path, 'selfdis_test.csv')):
        with open(os.path.join(opt.teachertest_path, 'selfdis_test.csv'), 'w') as f:
            f.write(','.join(vallog_headers) + '\n')

    return opt


def main():
    best_acc = 0
    best_acc_1_4 = 0
    best_acc_2_4 = 0
    best_acc_3_4 = 0
    best_acc_4_4 = 0
    best_acc_ens = 0
    best_epoch = 0

    opt = parse_option()
    if not os.path.exists(opt.save):
        os.makedirs(opt.save)

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

    def load_student(model_path, n_cls):
        print('==> loading teacher model')
        model_t = get_teacher_name(model_path)
        model_t = model_t
        model = model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(torch.load(model_path)['model'])
        print('==> done')
        return model
    def get_teacher_name(model_path):
        """parse teacher name"""
        segments = model_path.split('/')[-2].split('_')
        if segments[0] != 'wrn':
            return segments[0]
        else:
            return segments[0] + '_' + segments[1] + '_' + segments[2]
    def load_fc(model_path, n_cls):
        print('==> loading fc model')
        fclayer = SepAtt(num_classes=n_cls)
        fclayer.load_state_dict(torch.load(model_path)['model'])
        print('==> done')
        return fclayer
    def load_emb(model_path, n_cls):
        print('==> loading emb model')
        embedding = MetaEmbedding()
        embedding.load_state_dict(torch.load(model_path)['model'])
        print('==> done')
        return embedding

    model = model_dict[opt.model](num_classes=n_cls)

    fclayer = ReLayer1()
    embedding = MetaEmbedding()


    model_backbone = load_student(opt.path_backbone, n_cls)
    model_fc = load_fc(opt.path_fc, n_cls)



    #model_emb = load_emb(opt.path_emb, n_cls)

    
    # print("Model's state_dict:")
    # for param_tensor in model_fc.state_dict():
    #     print(param_tensor, "\t", model_fc.state_dict()[param_tensor].size())

    #model_fc.scala3.load_state_dict(model_fc.scala6.state_dict())

    trainable_list = nn.ModuleList([])
    trainable_list.append(model_backbone)
    trainable_list.append(model_fc)
    #trainable_list.append(model_emb)


    # loss
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = CosSimilarity()
    criterion_hint=HintLoss()

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_hint)

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    if torch.cuda.is_available():
        trainable_list.cuda()
        criterion_list = criterion_list.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)



    # routine

    dynamic_evaluate(trainable_list,val_loader,val_loader,opt)
    # top1_4_acc, top2_4_acc, top3_4_acc, top4_4_acc, topen_acc, test_acc_top5, test_loss,correctsum, errorsum, countone = \
    #         test(val_loader, trainable_list, criterion_list, opt)

    # print('correct similarity: ',  correctsum / 100)
    # print('error similarity: ', errorsum / 100)
    # print('increatment: ', countone / 100)
    # print('cos similarity: ', test_loss / 156)
        # top1_8_acc, top2_8_acc, top3_8_acc, top4_8_acc, top5_8_acc, top6_8_acc, top7_8_acc,top8_8_acc,topen_acc, test_acc_top5, test_loss = \
        #     validate(val_loader, trainable_list, criterion_list, opt)

        # save the best model



    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.





if __name__ == '__main__':
    main()
