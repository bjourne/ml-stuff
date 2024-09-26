Reimplementations of deep learning models.

* `ottt_cifar10.py`: From [Online Training Through Time for Spiking Neural Networks](https://arxiv.org/abs/2210.04195)

# Results

| DATASET  | MODEL    | VER           | BS  | AUG | ACC  |
|----------|----------|---------------|-----|-----|------|
| CIFAR10  | VGG16    | std           | ?   | std | 93.6 |
| CIFAR10  | VGG16    | lin-512-N_CLS | ?   | std | 92.1 |
| CIFAR10  | VGG16    | no-BN         | ?   | std | 92.6 |
| CIFAR10  | DenseNet | std           | 128 | std | 94.4 |
| CIFAR100 | VGG16    | std           | 256 | std | 70.7 |
| CIFAR100 | VGG16    | std           | 64  | std | 71.7 |
| CIFAR100 | VGG16    | std           | 256 | aa  | 74.7 |
| CIFAR100 | ResNet50 | std           | 256 | aa  | 47.7 |
| CIFAR100 | ResNet18 | std           | 256 | aa  | 56.1 |
