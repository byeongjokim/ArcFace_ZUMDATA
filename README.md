# ArcFace model using ZUM Data
Forked from [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)

# How to run
## Install
```
$pip install -r requirements.txt
```

## Configure
```
$vi config/config.py
```
- inf_json: json directory of zum data for preprocessing data
- root: face images directory
- total_list: total list of face images
- train_list: training list of face images
- val_list: validation list of face images
- test_list: testing list of face images
- val_pair_list: pair of validation list for verification accuracy
- zum_test_list: pair of testing list for verification accuracy
- backbone, max_epoch, lr ...: settings for training

## Preprocess
```
$python preprocess.py
```
Preprocessing **image, json** to **face image, list**

## Run
```
$python train.py
```

## Results