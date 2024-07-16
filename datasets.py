from typing import List

from torch import nn, Tensor, tensor
from torch.utils.data import Dataset
from torchvision import transforms, datasets

# list of all datasets
DATASETS = ["cifar10"]


def get_dataset(split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    return _cifar10(split)


def get_num_classes() -> int:
    """Return the number of classes in the dataset. """
    return 10


def get_normalize_layer() -> nn.Module:
    """Return the dataset's normalization layer"""
    return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)


_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())


class NormalizeLayer(nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
    and dividing by the dataset standard deviation.

    In order to certify radii in original coordinates rather than standardized coordinates, we
    add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
    layer of the classifier rather than as a part of preprocessing as is typical.
    """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.register_buffer('means', tensor(means))
        self.register_buffer('sds', tensor(sds))

    def forward(self, input_tensor: Tensor):
        device = input_tensor.device
        means = self.means.to(device)
        sds = self.sds.to(device)

        batch_size, num_channels, height, width = input_tensor.shape
        means = means.view(1, num_channels, 1, 1).expand(batch_size, num_channels, height, width)
        sds = sds.view(1, num_channels, 1, 1).expand(batch_size, num_channels, height, width)

        return (input_tensor - means) / sds
