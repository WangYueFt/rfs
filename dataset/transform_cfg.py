from __future__ import print_function

import numpy as np
from PIL import Image
import torchvision.transforms as transforms


mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
normalize = transforms.Normalize(mean=mean, std=std)


transform_A = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize
    ])
]


transform_B = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomResizedCrop(84, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize(92),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    ])
]

transform_C = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        # transforms.Resize(92, interpolation = PIL.Image.BICUBIC),
        transforms.RandomResizedCrop(80),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Lighting(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec']),
        # normalize
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize(92),
        transforms.CenterCrop(80),
        transforms.ToTensor(),
        # normalize
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    ])
]

# CIFAR style transformation
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]
normalize_cifar100 = transforms.Normalize(mean=mean, std=std)
transform_D = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ])
]


transforms_list = ['A', 'B', 'C', 'D']


transforms_options = {
    'A': transform_A,
    'B': transform_B,
    'C': transform_C,
    'D': transform_D,
}
