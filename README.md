Reimplementations of deep learning models.

* `ottt_cifar10.py`: From [Online Training Through Time for Spiking Neural Networks](https://arxiv.org/abs/2210.04195)

# Usage

    $ python main.py train vgg16 128 0.0005
    ...
    $ python main.py test vgg16 128 /tmp/logs/runs_vgg16_128_0.0005/net.pth
    ...
    $ python main.py --help
    ...


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
| PRG  | Training in progress       |
| N    | Number of epochs           |


| DATE       | DATASET  | MODEL        | VER     | BS    | AUG | SEED | WD     | DO  | N   | ACC  | PRG |
|------------|----------|--------------|---------|-------|-----|------|--------|-----|-----|------|-----|
|            | CIFAR10  | DenseNet     | std     | 128   | std |      | 0.0    |     |     | 94.9 | n   |
|            | CIFAR10  | DenseNet     | std     | 256   | std |      | 0.0    |     |     | 94.6 | n   |
|            | CIFAR10  | DenseNet     | std     | 512   | std |      | 0.0    |     |     | 93.8 | n   |
| 2025-02-11 | CIFAR10  | DenseNet     | std     | 256/8 | aa  | 1003 | 0.0    | 0.0 | 443 | 96.4 | n   |
|            | CIFAR10  | VGG16        | std     | ?     | std |      | 0.0    | 0.5 |     | 93.6 | n   |
|            | CIFAR10  | VGG16        | lin-512 | ?     | std |      | 0.0    | 0.5 |     | 92.1 | n   |
|            | CIFAR10  | VGG16        | no-BN   | ?     | std |      | 0.0    | 0.5 |     | 92.6 | n   |
| 2025-02-11 | CIFAR100 | DenseNet     | std     | 256/8 | aa  | 1007 | 0.0    | 0.0 | 415 | 78.4 | n   |
|            | CIFAR100 | VGG16        | std     | 32    | aa  | 1001 | 0.0    | 0.0 |     | 74.9 | n   |
|            | CIFAR100 | VGG16        | std     | 64    | aa  | 1001 | 0.0    | 0.5 |     | 69.1 | n   |
|            | CIFAR100 | VGG16        | std     | 64    | std |      | 0.0    | 0.5 |     | 71.7 | n   |
|            | CIFAR100 | VGG16        | std     | 128   | aa  | 1001 | 0.0    | 0.5 |     | 75.4 | n   |
| 2025-02-01 | CIFAR100 | VGG16        | std     | 128   | aa  | 1001 | 0.0005 | 0.0 |     | 77.4 | n   |
|            | CIFAR100 | VGG16        | std     | 256   | aa  |      | 0.0    | 0.5 |     | 74.7 | n   |
| 2024-10-02 | CIFAR100 | VGG16        | std     | 256   | aa  | 1001 | 0.0005 | 0.0 |     | 77.6 | n   |
| 2025-02-11 | CIFAR100 | VGG16        | std     | 256/8 | aa  | 1003 | 0.0004 | 0.0 | 600 | 74.3 | n   |
|            | CIFAR100 | VGG16        | std     | 256   | std |      | 0.0    | 0.5 |     | 70.7 | n   |
|            | CIFAR100 | VGG16        | std     | 512   | aa  | 1001 | 0.0    | 0.0 |     | 72.9 | n   |
| 2025-02-10 | CIFAR100 | VGG16        | std     | 512/8 | aa  | 1001 | 0.0005 | 0.0 |     | 73.4 | n   |
|            | CIFAR100 | VGG16QCFS    | std     | 128   | aa  | 1001 | 0.0    | 0.0 |     | 53.9 | n   |
|            | CIFAR100 | VGG16QCFS    | std     | 256   | aa  | 1001 | 0.0    | 0.5 |     | 72.0 | n   |
| 2025-02-11 | CIFAR100 | ResNet18     | std     | 512/8 | aa  | 1003 | 0.0005 | 0.0 | 555 | 77.3 | n   |
| 2025-02-11 | CIFAR100 | ResNet18     | std     | 256/8 | aa  | 1004 | 0.0    | 0.0 | 731 | 76.7 | y   |
|            | CIFAR100 | ResNet18     | std     | 256   | aa  |      | 0.0    |     |     | 59.1 | n   |
|            | CIFAR100 | ResNet50     | std     | 256   | aa  |      | 0.0    |     |     | 47.8 | n   |
|            | CIFAR100 | ResNet20     | std     | 256   | aa  | 1001 | 0.0    |     |     | 67.2 | n   |
| 2024-10-01 | CIFAR100 | ResNet20     | std     | 512   | aa  | 1001 | 0.0    | 0.0 |     | 69.1 | n   |
| 2024-10-01 | CIFAR100 | ResNet20     | std     | 256   | aa  | 1001 | 0.0    | 0.0 |     | 69.4 | n   |
| 2024-10-06 | CIFAR100 | ResNet18QCFS | std     | 256   | aa  | 1001 | 0.0005 | 0.0 |     | 79.8 | n   |
| 2024-10-06 | CIFAR100 | ResNet18QCFS | std     | 128   | aa  | 1001 | 0.0005 | 0.0 |     | 80.3 | n   |
|            |          |              |         |       |     |      |        |     |     |      |     |
