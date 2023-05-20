from .resnet import *
from .pytorch_pretrained_vit import ViT
from .resnext import *
from .vgg import *
from .densenet import *
import torch
import torchvision.models as models
import torchvision

def init_model(cfg):
    if cfg.MODEL.CLASSES == 10:
        if cfg.MODEL.PRETRAINED != True:
            if cfg.MODEL.NAME == 'resnext50':
                net = resnext50(10)
                return net
            if cfg.MODEL.NAME == 'resnet50':
                net = resnet50(10)
                return net
            if cfg.MODEL.NAME == 'resnet101':
                net = resnet101(10)
                return net
            if cfg.MODEL.NAME == 'resnet152':
                net = resnet152(10)
                return net
            if cfg.MODEL.NAME == 'vgg19':
                net = vgg19_bn(10)
                return net
        else:
            if cfg.MODEL.NAME == 'resnext50':
                net = models.resnext50_32x4d(pretrained=True)
                inc = net.fc.in_features
                net.fc = torch.nn.Linear(inc, cfg.MODEL.CLASSES)
                return net
            if cfg.MODEL.NAME == 'resnet50':
                net = models.resnet50(pretrained=True)
                inc = net.fc.in_features
                net.fc = torch.nn.Linear(inc, cfg.MODEL.CLASSES)
                return net
            if cfg.MODEL.NAME == 'resnet101':
                net = models.resnet101(pretrained=True)
                inc = net.fc.in_features
                net.fc = torch.nn.Linear(inc, cfg.MODEL.CLASSES)
                return net
            if cfg.MODEL.NAME == 'resnet152':
                net = models.resnet152(pretrained=True)
                inc = net.fc.in_features
                net.fc = torch.nn.Linear(inc, cfg.MODEL.CLASSES)
                return net
            if cfg.MODEL.NAME == 'vgg19':
                net = models.vgg19_bn(pretrained=True)
                inc = net.classifier[-1].in_features
                net.classifier[-1] = torch.nn.Linear(inc, cfg.MODEL.CLASSES)
                return net
            if cfg.MODEL.NAME == 'vit':
                return ViT('B_16_imagenet1k', pretrained=True, image_size=cfg.DATASET.IMAGESIZE, num_classes=cfg.MODEL.CLASSES)
    if cfg.MODEL.CLASSES == 100:
        if cfg.MODEL.PRETRAINED != True:
            if cfg.MODEL.NAME == 'resnet50':
                return resnet50(cfg.MODEL.CLASSES)
            if cfg.MODEL.NAME == 'resnet101':
                return resnet101(cfg.MODEL.CLASSES)
            if cfg.MODEL.NAME == 'resnet152':
                return resnet152(cfg.MODEL.CLASSES)
            if cfg.MODEL.NAME == 'resnext50':
                return resnext50(cfg.MODEL.CLASSES)
            if cfg.MODEL.NAME == 'vgg19':
                return vgg19_bn(cfg.MODEL.CLASSES)
            if cfg.MODEL.NAME == 'densenet161':
                return densenet161(cfg.MODEL.CLASSES)
            if cfg.MODEL.NAME == 'densenet121':
                return densenet121(cfg.MODEL.CLASSES)
            if cfg.MODEL.NAME == 'vit':
                return ViT('B_16_imagenet1k', pretrained=False, image_size=cfg.DATASET.IMAGESIZE, num_classes=cfg.MODEL.CLASSES)
        else:
            if cfg.MODEL.NAME == 'resnext50':
                net = models.resnext50_32x4d(pretrained=True)
                inc = net.fc.in_features
                net.fc = torch.nn.Linear(inc, cfg.MODEL.CLASSES)
                return net
            if cfg.MODEL.NAME == 'resnet50':
                net = models.resnet50(pretrained=True)
                inc = net.fc.in_features
                net.fc = torch.nn.Linear(inc, cfg.MODEL.CLASSES)
                return net
            if cfg.MODEL.NAME == 'resnet101':
                net = models.resnet101(pretrained=True)
                inc = net.fc.in_features
                net.fc = torch.nn.Linear(inc, cfg.MODEL.CLASSES)
                return net
            if cfg.MODEL.NAME == 'resnet152':
                net = models.resnet152(pretrained=True)
                inc = net.fc.in_features
                net.fc = torch.nn.Linear(inc, cfg.MODEL.CLASSES)
                return net
            if cfg.MODEL.NAME == 'vgg19':
                net = models.vgg19_bn(pretrained=True)
                inc = net.classifier[-1].in_features
                net.classifier[-1] = torch.nn.Linear(inc, cfg.MODEL.CLASSES)
                return net
            if cfg.MODEL.NAME == 'vit':
                return ViT('B_16_imagenet1k', pretrained=True, image_size=cfg.DATASET.IMAGESIZE, num_classes=cfg.MODEL.CLASSES)

    if cfg.MODEL.CLASSES == 1000:
        if cfg.MODEL.NAME == 'resnet18':
            return torchvision.models.resnet18(num_classes=cfg.MODEL.CLASSES)
        if cfg.MODEL.NAME == 'resnet50':
            net = models.resnet50(pretrained=cfg.MODEL.PRETRAINED)
            return net
        if cfg.MODEL.NAME == 'resnet101':
            net = models.resnet101(pretrained=cfg.MODEL.PRETRAINED)
            return net
        if cfg.MODEL.NAME == 'resnet152':
            net = models.resnet152(pretrained=cfg.MODEL.PRETRAINED)
            return net
        if cfg.MODEL.NAME == 'vgg19':
            net = models.vgg19_bn(pretrained=cfg.MODEL.PRETRAINED)
            return net
        if cfg.MODEL.NAME == 'resnext50':
            net = models.resnext50_32x4d(pretrained=cfg.MODEL.PRETRAINED)
            return net
    return
