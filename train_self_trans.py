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

from helper.util import adjust_learning_rate, accuracy, AverageMeter, LabelSmoothing, DisturbLabel, SCELoss, SPLLoss
from helper.loops import train_trans as train, validate_trans as validate
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss

from models.trans import SimilarityNet

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
    parser.add_argument('-b', '--beta', type=float, default=100.0, help='weight balance for other losses')
    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='80,130,180,210',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--ls_factor', default=0.1, type=float, help='smoothing factor')
    parser.add_argument('--cbls_decay', default=0.9, type=float)
    parser.add_argument('--growing_factor', default=1.1, type=float)

    # dataset
    parser.add_argument('--model', type=str, default='ResNet34-similarity',
                        choices=['ResNet2', 'ResNet18', 'ResNet34', 'ResNet50', 'resnet8', 'resnet14', 'resnet20',
                                 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet18-similarity',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'MobileNetV1', ])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'tiny-imagenet'],
                        help='dataset')

    parser.add_argument('-front_layer', help='Last layer of first model', type=str, default="layer2.3.relu")
    parser.add_argument('-end_layer', help='Last layer of second model', type=str, default="scala1.0.relu")
    parser.add_argument('-channel', help='Last layer of second model', type=int, default=128)

    parser.add_argument('-t', '--trial', type=int, default=98, help='the experiment id')

    parser.add_argument('--backbone_path', type=str,
                        default='./save/models/selfdis/ResNet34_cifar100_lr_0.05_decay_0.0005_trial_1/ResNet34_best.pth',
                        help='teacher model snapshot')
    parser.add_argument('--fc_path', type=str,
                        default='./save/models/selfdis/ResNet34_cifar100_lr_0.05_decay_0.0005_trial_1/fc_best.pth',
                        help='teacher model snapshot')
    parser.add_argument('--transform_path', type=str,
                        default='./save/models/selfdis/ResNet34_cifar100_lr_0.05_decay_0.0005_trial_1/transform46_best.pth',
                        help='teacher model snapshot')
    parser.add_argument('--init', choices=['random', 'perm', 'eye', 'ps_inv', 'ones-zeros'], default='random')

    opt = parser.parse_args()
    opt.teachertest_path = './result'

    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV1', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.001

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models/trans'
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
    if not os.path.exists(os.path.join(opt.teachertest_path, 'trans_test.csv')):
        with open(os.path.join(opt.teachertest_path, 'trans_test.csv'), 'w') as f:
            f.write(','.join(vallog_headers) + '\n')

    return opt


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

    model = SimilarityNet(opt.backbone_path, opt.fc_path, opt.transform_path, opt.front_layer, opt.end_layer,
                          opt.dataset,
                          n_cls, opt.channel, opt.init)
    # model.load_state_dict(torch.load('./save/models/selfdis/ResNet18_cifar100_lr_0.05_decay_0.0005_trial_0/ResNet18-similarity_best.pth')['model'])
    # print(model.front_model_fc.scala4[1].bn2.weight.data)
    # print(model.front_model_fc.scala4[1].bn2.bias.data)
    # print(model.front_model_fc.scala4[1].bn2.running_mean.data)
    # print(model.front_model_fc.scala4[1].bn2.running_var.data)
    # print(model.front_model_fc.scala4[1].bn2.num_batches_tracked.data)

    print("model", model)
    transformmodel46 = model.transform
    transformmodel46.train()

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # loss
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation

    if torch.cuda.is_available():
        model.cuda()
        criterion_list = criterion_list.cuda()
        cudnn.benchmark = True

    # routine
    for epoch in range(1, opt.epochs + 1):

        # adjust_learning_rate(epoch, opt, optimizer)

        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        scheduler.step()

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion_list, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

            transform46 = {
                'epoch': epoch,
                'model': transformmodel46.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }

            save_file_transform46 = os.path.join(opt.save_folder, '{}_best.pth'.format('transform46'))
            torch.save(transform46, save_file_transform46)
            print('saving the best model!')

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    with open(os.path.join(opt.teachertest_path, 'trans_test.csv'), 'a') as f:
        log = [opt.model, opt.trial, best_epoch, best_acc.item()]
        log = map(str, log)
        f.write(','.join(log) + '\n')


if __name__ == '__main__':
    main()
