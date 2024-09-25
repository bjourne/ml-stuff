# Code for optimal image augmentation.
#
# Inspired by: https://github.com/putshua/ANN_SNN_QCFS/blob/main/Preprocess/augment.py
from PIL import Image, ImageEnhance, ImageOps
from random import choice, random

import numpy as np
import torch

# code from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
# Improved Regularization of Convolutional Neural Networks with Cutout.
class Cutout2:
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

class SubPolicy(object):
    def __init__(self, p1, op1, idx1, p2, op2, idx2):
        self.p1 = p1
        self.p2 = p2
        fillcolor = 128, 128, 128

        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }
        funs = {
            "shearX": lambda img, mag: img.transform(
                img.size, Image.AFFINE, (1, mag *
                                         choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, mag: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, mag *
                                         choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, mag: img.transform(
                img.size, Image.AFFINE, (1, 0, mag *
                                         img.size[0] * choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, mag: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, mag *
                                         img.size[1] * choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, mag: img.rotate(mag, fillcolor = fillcolor),
            "color": lambda img, mag: ImageEnhance.Color(img).enhance(1 + mag * choice([-1, 1])),
            "posterize": lambda img, mag: ImageOps.posterize(img, mag),
            "solarize": lambda img, mag: ImageOps.solarize(img, mag),
            "contrast": lambda img, mag: ImageEnhance.Contrast(img).enhance(
                1 + mag * choice([-1, 1])),
            "sharpness": lambda img, mag: ImageEnhance.Sharpness(img).enhance(
                1 + mag * choice([-1, 1])),
            "brightness": lambda img, mag: ImageEnhance.Brightness(img).enhance(
                1 + mag * choice([-1, 1])),
            "autocontrast": lambda img, mag: ImageOps.autocontrast(img),
            "equalize": lambda img, mag: ImageOps.equalize(img),
            "invert": lambda img, mag: ImageOps.invert(img)
        }

        mag1 = ranges[op1][idx1]
        mag2 = ranges[op2][idx2]
        self.fun1 = lambda img: funs[op1](img, mag1)
        self.fun2 = lambda img: funs[op2](img, mag2)

    def __call__(self, img):
        if random() < self.p1:
            img = self.fun1(img)
        if random() < self.p2:
            img = self.fun2(img)
        return img

class ImageNetPolicy(object):
    def __init__(self):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8)
        ]

    def __call__(self, img):
        return choice(self.policies)(img)

class CIFARPolicy:
    def __init__(self):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1)
        ]

    def __call__(self, img):
        return choice(self.policies)(img)

def main():
    from torchvision.datasets import CIFAR10

    ds = CIFAR10('/tmp/data', True, download = True)
    x, y = ds[99]

    x.save('orig.png')
    x2 = rotate_with_fill(x, 45)
    x2.save('new.png')
    x3 = x.rotate(45, fillcolor = (128, 128, 128))
    x3.save('new2.png')

if __name__ == '__main__':
    main()
