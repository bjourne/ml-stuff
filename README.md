Reimplementations of deep learning models.

* `ottt_cifar10.py`: From [Online Training Through Time for Spiking Neural Networks](https://arxiv.org/abs/2210.04195)

# Results

| DATASET  | MODEL    | VER           | BS  | ACC  |
|----------|----------|---------------|-----|------|
| CIFAR10  | VGG16    | std           | ?   | 93.6 |
| CIFAR10  | VGG16    | lin-512-N_CLS | ?   | 92.1 |
| CIFAR10  | VGG16    | no-BN         | ?   | 92.6 |
| CIFAR10  | DenseNet | std           | 128 | 94.4 |
| CIFAR100 | VGG16    | std           | 256 | 70.7 |
| CIFAR100 | VGG16    | std           | 64  | 64.2 |
|          |          |               |     |      |
