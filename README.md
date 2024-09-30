Reimplementations of deep learning models.

* `ottt_cifar10.py`: From [Online Training Through Time for Spiking Neural Networks](https://arxiv.org/abs/2210.04195)

# Observations

* With dropout 0.5 VGG16 does not converge on CIFAR100 with learning rate 0.1 and batch
  size 32.

# Results

Abbreviations:

| ABBR | MEANING                    |
|------|----------------------------|
| ACC  | Validation accuracy        |
| AUG  | Data augmentation strategy |
| BS   | Batch size                 |
| DO   | Dropout                    |
| VER  | Version code               |
| WD   | Weight decay               |


| DATASET  | MODEL     | VER     | BS  | AUG | SEED | WD  | DO  | ACC  | PRG   |
|----------|-----------|---------|-----|-----|------|-----|-----|------|-------|
| CIFAR10  | VGG16     | std     | ?   | std |      | 0.0 | 0.5 | 93.6 |       |
| CIFAR10  | VGG16     | lin-512 | ?   | std |      | 0.0 | 0.5 | 92.1 |       |
| CIFAR10  | VGG16     | no-BN   | ?   | std |      | 0.0 | 0.5 | 92.6 |       |
| CIFAR10  | DenseNet  | std     | 128 | std |      | 0.0 |     | 94.4 |       |
| CIFAR100 | VGG16     | std     | 256 | std |      | 0.0 | 0.5 | 70.7 |       |
| CIFAR100 | VGG16     | std     | 64  | std |      | 0.0 | 0.5 | 71.7 |       |
| CIFAR100 | VGG16     | std     | 256 | aa  |      | 0.0 | 0.5 | 74.7 |       |
| CIFAR100 | ResNet50  | std     | 256 | aa  |      | 0.0 |     | 47.8 |       |
| CIFAR100 | ResNet18  | std     | 256 | aa  |      | 0.0 |     | 59.1 |       |
| CIFAR100 | ResNet20  | std     | 256 | aa  | 1001 | 0.0 |     | 67.2 |       |
| CIFAR100 | VGG16QCFS | std     | 256 | aa  | 1001 | 0.0 | 0.5 | 72.0 |       |
| CIFAR100 | VGG16     | std     | 64  | aa  | 1001 | 0.0 | 0.5 | 65.9 | y/srv |
| CIFAR100 | VGG16     | std     | 128 | aa  | 1001 | 0.0 | 0.5 | 74.6 | y/srv |
| CIFAR100 | VGG16QCFS | std     | 128 | aa  | 1001 | 0.0 | 0.0 | 53.9 | n     |
| CIFAR100 | VGG16     | std     | 32  | aa  | 1001 | 0.0 | 0.0 | 74.9 | n     |
|          |           |         |     |     |      |     |     |      |       |
