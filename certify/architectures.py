from typing import List

from torch import device, nn, tensor, cuda
from torch.backends import cudnn
from torchvision.models import resnet50

from certify.datasets import get_num_classes
from cifar_resnet import resnet

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def get_normalize_layer(dataset: str) -> nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)


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
        self.means = tensor(means).to(device="cuda" if cuda.is_available() else "cpu")
        self.sds = tensor(sds).to(device="cuda" if cuda.is_available() else "cpu")

    def forward(self, inputs: tensor):
        (batch_size, num_channels, height, width) = inputs.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (inputs - means) / sds


def get_architecture(dataset: str, torch_device: device) -> nn.Module:
    """ Return a neural network (with random weights)
    :param dataset: the dataset to use.
    :param torch_device: The device to which the model should be moved
    :return: a Pytorch module
    """
    if dataset == "imagenet":
        model = nn.DataParallel(resnet50(pretrained=False)).to(torch_device)
        cudnn.benchmark = True
    elif dataset == "cifar10":
        model = resnet(depth=110, num_classes=get_num_classes(dataset)).to(torch_device)
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")

    normalize_layer = get_normalize_layer(dataset).to(torch_device)
    return nn.Sequential(normalize_layer, model).to(torch_device)
