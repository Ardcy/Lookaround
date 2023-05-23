# Lookaround optimizer

This repository contains a PyTorch implementation of the Lookaround optimizer for DNNs from the paper Lookaround Optimizer: 
$k$ steps around, 1 step average.

# Usage

Due to the use of various data augmentation techniques in our approach, our optimizer needs to be coupled with a dataloader that supports the different data augmentations.

Here is an example using three data augmentation methods:

```python
from lookaround import Lookaround
optimizer = Lookaround(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, head_num=3, frequence=5) # k=5

train_iter1 = iter(trainloaders[0])
train_iter2 = iter(trainloaders[1])
train_iter3 = iter(trainloaders[2])
for batch_idx in range(len(trainloaders[0])):
    for batch in [train_iter1, train_iter2, train_iter3]:
        inputs, targets = next(batch)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
```

In our code, the invocation of different optimization methods is as follows:

```
python3 train.py --yaml_path=<YAML_PATH> \
                 --train_mode=<TRAIN_MODE> \
                 --cuda_id=<CUDA_ID> \
                 --data_dir=<DATASET> \
                 --out=<OUTPUT> \
                 --optimizer=<OPTIMIZER> 
```

Parameters:

* ```YAML_PATH``` &mdash; config yaml path to train
* ```TRAIN_MODE``` &mdash; optimization method [TRAIN_LOOKAROUND/TRAIN_SGD/TRAIN_SWA/TRAIN_LOOKAHEAD/TRAIN_ADAMW/TRAIN_SAM/TRAIN_SWAD]
* ```CUDA_ID``` &mdash; cuda device id
* ```DATASET``` &mdash; dataset folder
* ```OUTPUT``` &mdash; output foloder
* ```OPTIMIZER``` &mdash; optimizer name [Lookaround/SGD/Adam]

## Data

3 datasets were used in the paper:

* CIFAR-10
* CIFAR-100
* ImageNet: Downloadable from https://image-net.org/download.php

## Requirements

* PyTorch 1.8 or higher
* Python 3.6
  
To run SGG or other optimization method use the following command:

```
python train.py --yaml_path='resnet50_cifar10.yaml' --train_mode='TRAIN_SGD' --cuda_id=0 --data_dir='dataset/' --out='./out/' --optimizer='SGD'

python train.py --yaml_path='resnet50_cifar10.yaml' --train_mode='TRAIN_SWA' --cuda_id=0 --data_dir='dataset/' --out='./out/' --optimizer='SGD'

python train.py --yaml_path='resnet50_cifar10.yaml' --train_mode='TRAIN_LOOKAHEAD' --cuda_id=0 --data_dir='dataset/' --out='./out/' --optimizer='SGD'

python train.py --yaml_path='resnet50_cifar10.yaml' --train_mode='TRAIN_ADAMW' --cuda_id=0 --data_dir='dataset/' --out='./out/' 

python train.py --yaml_path='resnet50_cifar10.yaml' --train_mode='TRAIN_SAM' --cuda_id=0 --data_dir='dataset/' --out='./out/'  --optimizer='SGD'

python train.py --yaml_path='resnet50_cifar10.yaml' --train_mode='TRAIN_SWAD' --cuda_id=0 --data_dir='dataset/' --out='./out/' --optimizer='SGD'
```

To run Lookaround training use the following command:
```
python train.py --yaml_path='resnet50_cifar10.yaml' --train_mode='TRAIN_LOOKAROUND' --cuda_id=0 --data_dir='dataset/' --out='./out/' --optimizer='Lookaround'
```