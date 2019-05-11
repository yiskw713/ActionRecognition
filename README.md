# Action Recognition with pytorch

## Requirements
* python 3.x
* pytorch >= 1.0
* torchvision
* pandas
* numpy
* Pillow
* tqdm
* PyYAML
* addict
* tensorboardX
* adabound

## Dataset
### Kinetics

You can download videos in Kinetics with [the official donwloader](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).

Then you need to convert .mp4 files to .jpeg files.
You can do that using the code from [this repository](https://github.com/kenshohara/3D-ResNets-PyTorch/tree/work).

## Training
If you want to train a model, just run `python train.py ./PATH_TO_CONFIG_FILE`
For example, when running `python train.py ./result/resnet18/adabound/config.yaml`,
the configuration described in `./result/resnet18/adabound/config.yaml` will be used .

If you want to set your own configuration, please make config.yaml like this:
```
model: resnet18
msc: False            # if you use temporal multi-scale input

class_weight: True    # if you use class weight to calculate cross entropy or not
writer_flag: True     # if you use tensorboardx or not

n_classes: 400
batch_size: 32
input_frames: 16
height: 224
width: 224
num_workers: 8
max_epoch: 200

optimizer: AdaBound
learning_rate: 0.001
lr_patience: 10       # Patience of LR scheduler
momentum: 0.9         # momentum of SGD
dampening: 0.0        # dampening for momentum of SGD
weight_decay: 0.0001  # weight decay
nesterov: True        # enables Nesterov momentum
final_lr: 0.1         # final learning rate for AdaBound

dataset_dir: /xxxx/xxxx/xxxxx/xxxxx
train_csv: ./dataset/train.csv
val_csv: ./dataset/val.csv
result_path: ./result/resnet18/adabound
```

## References
* [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)
* [SlowFastNetworks](https://github.com/r1ch88/SlowFastNetworks)
