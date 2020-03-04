import sys

import numpy

import torch

#from dataset import CIFAR100Train, CIFAR100Test

def networks(networks, use_gpu=True):
    """ return given network
    """

    if networks == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif networks == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif networks == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif networks == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif networks == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif networks == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif networks == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif networks == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif networks == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif networks == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif networks == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif networks == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif networks == 'xception':
        from models.xception import xception
        net = xception()
    elif networks == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif networks == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif networks == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif networks == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif networks == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif networks == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif networks == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif networks == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif networks == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif networks == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif networks == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif networks == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif networks == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif networks == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif networks == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif networks == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif networks == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif networks == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif networks == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif networks == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif networks == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif networks == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif networks == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif networks == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif networks == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif networks == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    # if use_gpu:
    #     net = net.cuda()

    return net
