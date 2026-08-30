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
from helper.loops import train_feature as train, val_feature as validate
from models.util import FcLayer, Adaptation_layers, ReLayer1,simfc,Transform,Scala1,Scala4,Scala6
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss


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

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')
    parser.add_argument('--path_t', type=str, default='./save/models/ResNet18_vanilla/ckpt_epoch_240.pth',
                        help='teacher model snapshot')
    parser.add_argument('--backbone_path', type=str,
                        default='./save/models/selfdis/ResNet34_cifar100_lr_0.05_decay_0.0005_trial_1/ResNet34_best.pth',
                        help='teacher model snapshot')
    parser.add_argument('--fc_path', type=str,
                        default='./save/models/selfdis/ResNet34_cifar100_lr_0.05_decay_0.0005_trial_1/fc_best.pth',
                        help='teacher model snapshot')
    parser.add_argument('--tran_path46', type=str,
                        default='./save/models/selfdis/ResNet34_cifar100_lr_0.05_decay_0.0005_trial_1/transform46_best.pth',
                        help='teacher model snapshot')
    parser.add_argument('--tran_path14', type=str,
                        default='./save/models/selfdis/ResNet34_cifar100_lr_0.05_decay_0.0005_trial_1/transform14_best.pth',
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
    model_t = model_t + 'Sim'
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model

def load_fc(model_path):
    print('==> loading fc model')
    fclayer = simfc(512,100)
    fclayer.load_state_dict(torch.load(model_path)['model'],strict=False)
    print('==> done')
    return fclayer

def load_reduction1(model_path):
    print('==> loading fc model')
    relayer1 = Scala1()
    relayer1.load_state_dict(torch.load(model_path)['model'],strict=False)
    print('==> done')
    return relayer1

def load_reduction2(model_path):
    print('==> loading fc model')
    relayer2 = Scala4()
    relayer2.load_state_dict(torch.load(model_path)['model'],strict=False)
    print('==> done')
    return relayer2

def load_reduction3(model_path):
    print('==> loading fc model')
    relayer3 = Scala6()
    relayer3.load_state_dict(torch.load(model_path)['model'],strict=False)
    print('==> done')
    return relayer3

def load_tran1(model_path):
    print('==> loading fc model')
    transform14 = Transform(128, 128)
    transform14.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return transform14
def load_tran2(model_path):
    print('==> loading fc model')
    transform46 = Transform(256, 256)
    transform46.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return transform46
def main():
    best_acc = 0
    best_epoch = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = load_teacher(opt.backbone_path, n_cls)
    print("model_t", model)

    relayer1 = load_reduction1(opt.fc_path)
    print("fclayer", relayer1)

    relayer2 = load_reduction2(opt.fc_path)
    print("fclayer", relayer2)

    relayer3 = load_reduction3(opt.fc_path)
    print("fclayer", relayer3)

    transform14 = load_tran1(opt.tran_path14)
    print("transform14", transform14)

    transform46 = load_tran2(opt.tran_path46)
    print("transform46", transform46)

    fclayer = load_fc(opt.fc_path)
    print("fclayer", fclayer)

    trainable_list = nn.ModuleList([])
    trainable_list.append(model)
    trainable_list.append(relayer1)
    trainable_list.append(relayer2)
    trainable_list.append(relayer3)
    trainable_list.append(transform14)
    trainable_list.append(transform46)
    trainable_list.append(fclayer)

    # loss
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation

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



    test_acc,  test_loss = validate(val_loader, trainable_list, criterion_list, opt)



    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', test_acc)
    with open(os.path.join(opt.teachertest_path, 'feature_test.csv'), 'a') as f:
        log = [opt.model,opt.trial,best_epoch, test_acc.item()]
        log = map(str, log)
        f.write(','.join(log) + '\n')


if __name__ == '__main__':
    main()
