## Metal Artifact Reduction in CT Image

### Dependenices
* (More detail See [mar.yaml](mar.yaml))
A suitable [conda](https://conda.io/) environment named `mar` can be created and activated with:

```
conda env create -f mar.yaml
conda activate mar
```

### Dataset Preparation
Download the training dataset train_640geo by [[BaiduYun]](https://pan.baidu.com/s/1d21KQUDlp8OjxGA0JfNDlw) (Password: i6ec) and put them into [./data/train/](./data/train/).

Download the testing dataset test_640geo by [[BaiduYun]](https://pan.baidu.com/s/1EC2rgU4MWPt6PI9t5PHZ-g) (Password: 5ta7) and put them into [./data/test/](./data/test/).

Training and testing datasets can also available here [[BaiduYun]](https://pan.baidu.com/s/1Tu-vTrx7OYCr7eoRDoAgDw?pwd=dicd#list/path=%2F) (Password:dicd).

### Train
```
cd xxx
python train.py --gpu_id 0
```

### Test
```
cd xxx
python test.py --gpu_id 0
```
