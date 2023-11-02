## Super-Resolution Experiments

### Dependenices

* python3
* pytorch >= 1.6
* NVIDIA GPU + CUDA
* Python packages: pip3 install numpy opencv-python lmdb pyyaml

### Quick Start
```bash
cd codes/config/KXNet
python3 test.py -opt ./options/setting2/test/test_setting2_x2_EX.yml
```

## Pretrained models
Pretrained models of KXNet_EX are available as [here](https://github.com/jiahong-fu/Equivariant-Proximal-Operator/releases/tag/v1.0). Download the models to [./checkpoints](./checkpoints)

### Dataset Preparation
For train
We use [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) as our training datasets. Download them and put the images in [DIV2K_Flickr2K](./codes/data/DIV2K_Flickr2K).

For evaluation, we use four datasets, i.e., [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip), [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip), [Urban100](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip), and [BSD100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip). All test datesets can be downloaded from [BaiduYun](https://pan.baidu.com/s/1ShoqALXdzsELDKPbexNFtQ) (Password: uag1).

**Setting**: We separately set the kernel width as $\lambda_1 = 0.8, \lambda_2 = 1.6$ and $\lambda_1 = 2.0, \lambda_2 = 4.0$, and rotate them by $\theta \in { 0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4} }$, respectively. This means we have 8 different blur kernels and use them to generate test data at noise level $\sigma \in { 0, 5, 15}$. Run ['codes/scripts/generate_mod_anisoblur_LR_sfold.py'](codes/scripts/generate_mod_anisoblur_LR_sfold.py') to generate LRblur/HR datasets.

### Train

For single GPU:
```bash
cd codes/config/KXNet
python3 train.py -opt=options/setting2/train/train_setting2_x2_EX.yml
```

For distributed training
```bash
cd codes/config/KXNet
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train.py -opt=options/setting2/train/train_setting2_x2_EX.yml --launcher pytorch
```

### Test on Synthetic Images
```bash
cd codes/config/KXNet
python3 test.py -opt=options/setting2/test/test_setting2_x2_EX.yml
```
