from torch import device, nn

from cifar_resnet import resnet
from datasets import get_normalize_layer, get_num_classes

# cifar_resnet20 - a 20-layer residual network sized for CIFAR10
ARCHITECTURES = ["cifar_resnet20"]


def get_architecture(torch_device: device) -> nn.Module:
    """ Return a neural network (with random weights)
    :param torch_device: The device to which the model should be moved
    :return: a Pytorch module
    """
    model = resnet(depth=110, num_classes=get_num_classes()).to(torch_device)
    normalize_layer = get_normalize_layer().to(torch_device)
    return nn.Sequential(normalize_layer, model).to(torch_device)
