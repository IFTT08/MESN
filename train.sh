
CUDA_VISIBLE_DEVICES=0 python train_backbone.py --model ResNet18 --dataset cifar100 -r 1  --trial 0
CUDA_VISIBLE_DEVICES=0 python train_menet.py --model ResNet18 --dataset cifar100 -r 1  --trial 0
