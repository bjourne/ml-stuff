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


| DATE     | DATASET  | MODEL     | VER     | BS  | AUG | SEED | WD     | DO  | ACC  | PRG   |
|----------|----------|-----------|---------|-----|-----|------|--------|-----|------|-------|
|          | CIFAR10  | VGG16     | std     | ?   | std |      | 0.0    | 0.5 | 93.6 | n     |
|          | CIFAR10  | VGG16     | lin-512 | ?   | std |      | 0.0    | 0.5 | 92.1 | n     |
|          | CIFAR10  | VGG16     | no-BN   | ?   | std |      | 0.0    | 0.5 | 92.6 | n     |
|          | CIFAR10  | DenseNet  | std     | 128 | std |      | 0.0    |     | 94.4 | n     |
|          | CIFAR100 | VGG16     | std     | 256 | std |      | 0.0    | 0.5 | 70.7 | n     |
|          | CIFAR100 | VGG16     | std     | 64  | std |      | 0.0    | 0.5 | 71.7 | n     |
|          | CIFAR100 | VGG16     | std     | 256 | aa  |      | 0.0    | 0.5 | 74.7 | n     |
|          | CIFAR100 | ResNet50  | std     | 256 | aa  |      | 0.0    |     | 47.8 | n     |
|          | CIFAR100 | ResNet18  | std     | 256 | aa  |      | 0.0    |     | 59.1 | n     |
|          | CIFAR100 | ResNet20  | std     | 256 | aa  | 1001 | 0.0    |     | 67.2 | n     |
|          | CIFAR100 | VGG16QCFS | std     | 256 | aa  | 1001 | 0.0    | 0.5 | 72.0 | n     |
|          | CIFAR100 | VGG16     | std     | 64  | aa  | 1001 | 0.0    | 0.5 | 67.6 | y/srv |
|          | CIFAR100 | VGG16     | std     | 128 | aa  | 1001 | 0.0    | 0.5 | 75.2 | y/srv |
|          | CIFAR100 | VGG16QCFS | std     | 128 | aa  | 1001 | 0.0    | 0.0 | 53.9 | n     |
|          | CIFAR100 | VGG16     | std     | 32  | aa  | 1001 | 0.0    | 0.0 | 74.9 | n     |
|          | CIFAR100 | VGG16     | std     | 512 | aa  | 1001 | 0.0    | 0.0 | 71.0 | y/srv |
| 20241001 | CIFAR100 | ResNet20  | std     | 512 | aa  | 1001 | 0.0    | 0.0 | 69.1 | n     |
| 20241001 | CIFAR100 | ResNet20  | std     | 256 | aa  | 1001 | 0.0    | 0.0 | 69.4 | n     |
| 20241002 | CIFAR100 | VGG16     | std     | 256 | aa  | 1001 | 0.0005 | 0.0 | 62.4 | y/dsk |
|          |          |           |         |     |     |      |        |     |      |       |
