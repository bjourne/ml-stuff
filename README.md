Reimplementations of deep learning models.

* `ottt_cifar10.py`: From [Online Training Through Time for Spiking Neural Networks](https://arxiv.org/abs/2210.04195)

# Usage

    $ python main.py --help
    ...
    $ python main.py --batch-size 128 vgg16 cifar100 train /tmp/logs
    ...
    $ python main.py --batch-size 128 vgg16 cifar100 test /path/to/net.pth
    ...
    $ CUDA_VISIBLE_DEVICES=-1 python main.py --batch-size 128 vgg16qcfs cifar100 train /tmp/logs

# Observations

* With dropout 0.5 VGG16 does not converge on CIFAR100 with learning
  rate 0.1 and batch size 32.

# Results

Abbreviations:

| ABBR | MEANING                    |
|------|----------------------------|
| ACC  | Validation accuracy        |
| AUG  | Data augmentation strategy |
| BS   | Batch size                 |
| DO   | Dropout                    |
| LR   | Learning rate              |
| VER  | Version code               |
| WD   | Weight decay               |
| PRG  | Training in progress       |
| N    | Number of epochs           |


| DATE       | DATASET  | MODEL        | VER     | BS     | AUG | WD     | DO  | LR   | SEED | N   | ACC  | PRG |
|------------|----------|--------------|---------|--------|-----|--------|-----|------|------|-----|------|-----|
| 2025-02-19 | CIFAR10  | AlexNet      | std     | 128/8  | std | 0.0    | 0.0 | 0.1  | 1000 | 976 | 94.3 | n   |
|            | CIFAR10  | DenseNet     | std     | 128    | std | 0.0    |     |      |      |     | 94.9 | n   |
|            | CIFAR10  | DenseNet     | std     | 256    | std | 0.0    |     |      |      |     | 94.6 | n   |
|            | CIFAR10  | DenseNet     | std     | 512    | std | 0.0    |     |      |      |     | 93.8 | n   |
| 2025-02-11 | CIFAR10  | DenseNet     | std     | 256/8  | aa  | 0.0    | 0.0 |      | 1003 | 443 | 96.4 | n   |
|            | CIFAR10  | VGG16        | std     | ?      | std | 0.0    | 0.5 |      |      |     | 93.6 | n   |
|            | CIFAR10  | VGG16        | lin-512 | ?      | std | 0.0    | 0.5 |      |      |     | 92.1 | n   |
|            | CIFAR10  | VGG16        | no-BN   | ?      | std | 0.0    | 0.5 |      |      |     | 92.6 | n   |
| 2025-02-11 | CIFAR100 | DenseNet     | std     | 256/8  | aa  | 0.0    | 0.0 |      | 1007 | 415 | 78.4 | n   |
| 2025-02-15 | CIFAR100 | EffNet-B0    | std     | 256/8  | aa  | 0.0    | 0.0 |      | 1010 | 201 | 75.5 | n   |
| 2025-02-15 | CIFAR100 | EffNet-B0    | std     | 512/8  | aa  | 0.0    | 0.0 |      | 1010 | 516 | 77.4 | n   |
| 2025-02-18 | CIFAR100 | ResNet18     | std     | 128/8  | aa  | 0.0000 | 0.0 | 0.10 | 1000 | 710 | 76.8 | n   |
| 2025-02-11 | CIFAR100 | ResNet18     | std     | 256/8  | aa  | 0.0    | 0.0 |      | 1004 | 731 | 76.7 | n   |
| 2025-02-14 | CIFAR100 | ResNet18     | std     | 256/8  | aa  | 0.0004 | 0.0 |      | 1012 | 908 | 77.1 | n   |
| 2025-02-14 | CIFAR100 | ResNet18     | std     | 256/8  | aa  | 0.0005 | 0.0 |      | 1010 | 765 | 77.7 | n   |
| 2025-02-14 | CIFAR100 | ResNet18     | std     | 256/8  | aa  | 0.0005 | 0.0 |      | 1011 | 923 | 77.7 | n   |
| 2025-02-14 | CIFAR100 | ResNet18     | std     | 256/8  | aa  | 0.0010 | 0.0 |      | 1015 | 637 | 75.0 | n   |
| 2025-02-13 | CIFAR100 | ResNet18     | std     | 512/8  | aa  | 0.0    | 0.0 |      | 1010 | 636 | 76.2 | n   |
| 2025-02-11 | CIFAR100 | ResNet18     | std     | 512/8  | aa  | 0.0005 | 0.0 |      | 1003 | 555 | 77.3 | n   |
| 2025-02-11 | CIFAR100 | ResNet18     | std     | 512/8  | aa  | 0.0010 | 0.0 |      | 1010 | 619 | 76.8 | n   |
| 2025-02-12 | CIFAR100 | ResNet18     | std     | 1024/8 | aa  | 0.0    | 0.0 |      | 1010 | 572 | 75.6 | n   |
| 2025-02-12 | CIFAR100 | ResNet18     | std     | 1024/8 | aa  | 0.0010 | 0.0 |      | 1010 | 771 | 77.2 | n   |
| 2025-02-12 | CIFAR100 | ResNet18     | std     | 1024/8 | aa  | 0.0020 | 0.0 |      | 1010 | 554 | 74.2 | n   |
|            | CIFAR100 | ResNet50     | std     | 256    | aa  | 0.0    |     |      |      |     | 47.8 | n   |
|            | CIFAR100 | ResNet20     | std     | 256    | aa  | 0.0    |     |      | 1001 |     | 67.2 | n   |
| 2024-10-01 | CIFAR100 | ResNet20     | std     | 512    | aa  | 0.0    | 0.0 |      | 1001 |     | 69.1 | n   |
| 2024-10-01 | CIFAR100 | ResNet20     | std     | 256    | aa  | 0.0    | 0.0 |      | 1001 |     | 69.4 | n   |
| 2025-02-17 | CIFAR100 | ResNet20     | std     | 128/8  | aa  | 0.0005 | 0.0 | 0.10 | 1000 | 953 | 72.8 | n   |
| 2025-02-18 | CIFAR100 | ResNet20-PA  | std     | 128/8  | aa  | 0.0005 | 0.0 | 0.05 | 1000 | 962 | 71.6 | n   |
| 2025-02-18 | CIFAR100 | ResNet20-PA  | std     | 128/8  | aa  | 0.0005 | 0.0 | 0.10 | 1000 | 975 | 71.7 | n   |
| 2025-02-18 | CIFAR100 | ResNet20-PA  | std     | 128/8  | aa  | 0.0005 | 0.0 | 0.10 | 1001 | 739 | 71.0 | n   |
| 2024-10-06 | CIFAR100 | ResNet18QCFS | std     | 256    | aa  | 0.0005 | 0.0 |      | 1001 |     | 79.8 | n   |
| 2024-10-06 | CIFAR100 | ResNet18QCFS | std     | 128    | aa  | 0.0005 | 0.0 |      | 1001 |     | 80.3 | n   |
|            | CIFAR100 | VGG16        | std     | 32     | aa  | 0.0    | 0.0 |      | 1001 |     | 74.9 | n   |
|            | CIFAR100 | VGG16        | std     | 64     | aa  | 0.0    | 0.5 |      | 1001 |     | 69.1 | n   |
|            | CIFAR100 | VGG16        | std     | 64     | std | 0.0    | 0.5 |      |      |     | 71.7 | n   |
|            | CIFAR100 | VGG16        | std     | 128    | aa  | 0.0    | 0.5 |      | 1001 |     | 75.4 | n   |
| 2025-02-01 | CIFAR100 | VGG16        | std     | 128    | aa  | 0.0005 | 0.0 |      | 1001 |     | 77.4 | n   |
|            | CIFAR100 | VGG16        | std     | 256    | aa  | 0.0    | 0.5 |      |      |     | 74.7 | n   |
| 2024-10-02 | CIFAR100 | VGG16        | std     | 256    | aa  | 0.0005 | 0.0 |      | 1001 |     | 77.6 | n   |
| 2025-02-11 | CIFAR100 | VGG16        | std     | 256/8  | aa  | 0.0004 | 0.0 |      | 1003 | 600 | 74.3 | n   |
|            | CIFAR100 | VGG16        | std     | 256    | std | 0.0    | 0.5 |      |      |     | 70.7 | n   |
|            | CIFAR100 | VGG16        | std     | 512    | aa  | 0.0    | 0.0 |      | 1001 |     | 72.9 | n   |
| 2025-02-10 | CIFAR100 | VGG16        | std     | 512/8  | aa  | 0.0005 | 0.0 |      | 1001 |     | 73.4 | n   |
|            | CIFAR100 | VGG16QCFS    | std     | 128    | aa  | 0.0    | 0.0 |      | 1001 |     | 53.9 | n   |
|            | CIFAR100 | VGG16QCFS    | std     | 256    | aa  | 0.0    | 0.5 |      | 1001 |     | 72.0 | n   |
| 2025-03-11 | CIFAR100 | VGG16QCFS    | std     | 256/8  | aa  | 0.0005 | 0.0 | 0.05 | 999  | 984 | 76.1 | n   |
| 2025-02-17 | CIFAR100 | VGG16QCFS    | std     | 256/8  | aa  | 0.0005 | 0.0 | 0.05 | 1010 | 963 | 76.0 | n   |
| 2025-02-17 | CIFAR100 | VGG16QCFS    | std     | 256/8  | aa  | 0.0005 | 0.0 | 0.10 | 1010 | 966 | 76.4 | n   |
| 2025-02-17 | CIFAR100 | VGG16QCFS    | std     | 256/8  | aa  | 0.0005 | 0.0 | 0.20 | 1010 | 703 | 72.6 | n   |
|            |          |              |         |        |     |        |     |      |      |     |      |     |
