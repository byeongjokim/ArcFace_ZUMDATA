# ArcFace model using ZUM Data
Forked from [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)

# How to run
## Install
### Dependency
```
$pip install -r requirements.txt
```

### Data
```
download https://drive.google.com/drive/folders/1erbIN37Z1QLNFnO9rnwbu9wNoCpc0HjA?usp=sharing
unzip ZUM
```

### Checkpoints
```
download https://drive.google.com/drive/folders/1erbIN37Z1QLNFnO9rnwbu9wNoCpc0HjA?usp=sharing
unzip checkpoints.zip
```

File Structure
- data/
- models/
- checkpoints/
- AlignDlib.py
- preprocess.py
- test.py
- train.py

## Configure
- backbone: resnet18/resnet50
- inf_json: json directory of zum data for preprocessing data
- root: face images directory
- total_list: total list of face images
- train_list: training list of face images
- val_list: validation list of face images
- test_list: testing list of face images
- val_pair_list: pair of validation list for verification accuracy
- zum_test_list: pair of testing list for verification accuracy
- max_epoch, lr ...: settings for training

## Preprocess
```
$python preprocess.py
```
Preprocessing **image, json** to **face image, list**

## Train
```
$python train.py [options]
```

## Test
```
$python test_only.py [options]
```

## Results (Verification Accuracy)
- resnet50: 87.5%
