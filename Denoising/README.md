## Denoising Experiments

### Dependenices
* python3
* pytorch >= 1.6
* NVIDIA GPU + CUDA
* Python packages: pip3 install numpy opencv-python lmdb pyyaml pytorch-ssim scipy matplotlib scikit-learn

### Quick Start for Testing
```
python main_test.py --model resnet_fconv --scale 1 --pre_train ../experiment/FConv/model/model_best.pt --save ../experiment/FConv/ --kernel_size 5 --n_resblocks 16 --n_feats 32 --res_scale 0.1 --tranNum 8 --device 0 --data_test Set5
```

### Pretrained Models
Pretrained Models are available as [[BaiduYun]](https://pan.baidu.com/s/1UIc4YB8KhaZLH_z_4TgXhg) (Password: wqip). Download the Pretrained models and put them into [./experiment/XXX/model/](./experiment/)
### Dataset Preparation
* **Training Dataset:**
Download the training data [[BaiduYun]](https://pan.baidu.com/s/1kDFFnU78mAszUZdfeqzcNQ) (Password: ssr9) and put them into the corresponding files [./Data/DIV2K/DIV2K_train_HR/](./Data/DIV2K/DIV2K_train_HR/) and [./Data/DIV2K/DIV2K_train_Noised/N1/](./Data/DIV2K/DIV2K_train_Noised/N1/).

* **Testing Dataset:**
Download the testing data [[BaiduYun]](https://pan.baidu.com/s/1Iz2m8RMSQaC7aeQEnPv2Cg) (Password: 8q97) and put them into the corresponding files [./Data/benchmark/](./Data/benchmark/)

### Train and Test
The bash commands for training and testing are provided in [./src/run.sh](./src/run.sh)
