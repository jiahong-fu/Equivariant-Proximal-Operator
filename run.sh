#!/bin/bash

# Denoising Experiments for Proximal Operator
# Training
# CNN
python main.py --model resnet_cnn --scale 1 --save CNN --n_resblocks 16 --n_feats 256 --res_scale 0.1 --epoch 150 --decay 100 --patch_size 48  --device 0
# SCUNet
python main.py --model scunet --scale 1 --save SCUNet --epoch 150 --decay 100 --patch_size 48  --device 0
# G-CNN
python main.py --model resnet_gcnn --scale 1 --save G-CNN --n_resblocks 16 --n_feats 64 --res_scale 0.1 --tranNum 4 --epoch 150 --decay 100 --patch_size 48 --device 0
# E2-CNN
python main.py --model resnet_e2cnn --scale 1 --save E2-CNN --n_resblocks 16 --n_feats 32 --res_scale 0.1 --tranNum 8 --epoch 150 --decay 100 --patch_size 48 --device 0
# PDO-eConv
python main.py --model resnet_pdoe --scale 1 --save PDO-eConv --n_resblocks 16 --n_feats 32 --res_scale 0.1 --tranNum 8 --epoch 150 --decay 100 --patch_size 48 --lr 5e-4 --device 0
# FConv
python main.py --model resnet_fconv --scale 1 --save FConv --n_resblocks 16 --n_feats 32 --res_scale 0.1 --tranNum 8 --epoch 150 --decay 100 --patch_size 48 --lr 5e-4 --device 0

# Testing
# CNN
python main_test.py --model resnet_cnn --scale 1 --pre_train ../experiment/CNN/model/model_best.pt --save ../experiment/CNN/ --kernel_size 5 --n_resblocks 16 --n_feats 256 --res_scale 0.1 --device 0
# SCUNet
python main_test.py --model scunet --scale 1 --pre_train ../experiment/SCUNet/model/model_best.pt --save ../experiment/SCUNet/ --device 0
# G-CNN
python main_test.py --model resnet_gcnn --scale 1 --pre_train ../experiment/G-CNN/model/model_best.pt --save ../experiment/G-CNN/ --kernel_size 5 --n_resblocks 16 --n_feats 64 --res_scale 0.1 --tranNum 4 --device 0
# E2-CNN
python main_test.py --model resnet_e2cnn --scale 1 --pre_train ../experiment/E2-CNN/model/model_best.pt --save ../experiment/E2-CNN/ --kernel_size 5 --n_resblocks 16 --n_feats 32 --res_scale 0.1 --tranNum 8 --device 0
# PDO-eConv
python main_test.py --model resnet_pdoe --scale 1 --pre_train ../experiment/PDO-eConv/model/model_best.pt --save ../experiment/PDO-eConv/ --kernel_size 5 --n_resblocks 16 --n_feats 32 --res_scale 0.1 --tranNum 8 --device 0
# FConv
python main_test.py --model resnet_fconv --scale 1 --pre_train ../experiment/FConv/model/model_best.pt --save ../experiment/FConv/ --kernel_size 5 --n_resblocks 16 --n_feats 32 --res_scale 0.1 --tranNum 8 --device 0