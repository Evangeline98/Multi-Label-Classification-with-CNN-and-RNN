from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as M

from utils import ON_KAGGLE


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def create_net(net_cls, pretrained: bool):
    if ON_KAGGLE and pretrained:
        net = net_cls()
        model_name = net_cls.__name__
        weights_path = f'../input/{model_name}/{model_name}.pth'
        net.load_state_dict(torch.load(weights_path))
    else:
        #net = net_cls(pretrained=pretrained)
        net = net_cls()
        model_name = net_cls.__name__
        net.load_state_dict(torch.load(f'{model_name}.pth'))
        print(model_name)

    return net

class ResNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes)
                )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)




    def fresh_params(self):
        return self.net.fc.parameters()


    def forward(self, x):
        return self.net(x)

class ResNet0(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes)
                )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

        self.net.fc1 = nn.Linear(num_classes,num_classes)
        self.init_weight()



    def fresh_params(self):
        return self.net.fc.parameters()

    def init_weight(self):
        weight = torch.load('/nfsshare/home/white-hearted-orange/data/sim.pt')
        self.net.fc1.weight = nn.Parameter(weight)


    def forward(self, x):
        return self.net(x)

class ResNet1(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes)
                )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

        self.net.fc1 = nn.Linear(num_classes,300)
        self.net.fc2 = nn.Linear(300, num_classes)

        self.init_weight()



    def fresh_params(self):
        return self.net.fc.parameters()

    def init_weight(self):
        weight = torch.load('/nfsshare/home/white-hearted-orange/data/em.pt')
        self.net.fc1.weight = nn.Parameter(weight)


    def forward(self, x):
        return self.net(x)



class DenseNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.densenet121):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.avg_pool = AvgPool()
        self.net.classifier = nn.Linear(
            self.net.classifier.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.net.classifier(out)
        return out

resnet50 = partial(ResNet, net_cls = M.resnet50)
resnet50em0 = partial(ResNet0, net_cls=M.resnet50)
resnet50em1 = partial(ResNet1, net_cls = M.resnet50)

