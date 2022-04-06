"""The models subpackage contains definitions for the following model
architectures:
-  `ResNeXt` for CIFAR10 CIFAR100
You can construct a model with random weights by calling its constructor:
.. code:: python
    import models
    resnext29_16_64 = models.ResNeXt29_16_64(num_classes)
    resnext29_8_64 = models.ResNeXt29_8_64(num_classes)
    resnet20 = models.ResNet20(num_classes)
    resnet32 = models.ResNet32(num_classes)


.. ResNext: https://arxiv.org/abs/1611.05431
"""

from .preresnet import preresnet20, preresnet32, preresnet44, preresnet56, preresnet110
from .caffe_cifar import caffe_cifar
from .resnet_mod import resnet_mod20, resnet_mod32, resnet_mod44, resnet_mod56, resnet_mod110

from .imagenet_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .ResNet_GN import resnet18_gn, resnet34_gn, resnet50_gn, resnet101_gn, resnet152_gn
from .BagNet import bagnet9, bagnet17, bagnet33

from .resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from .resnext import resnext29_8_64, resnext29_16_64
from .VGG import vgg16_bn, vgg16
from .googlenet import googlenet
from .densenet import DenseNet121, DenseNet169, DenseNet201, DenseNet161, densenet_cifar
from .dpn import dpn26, dpn92
from .senet import senet18
from .wrn import wrn
from .mnistnet import mnistnet