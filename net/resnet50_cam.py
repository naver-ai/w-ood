# Original Code: https://github.com/jiwoon-ahn/irn

import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50


class Net(nn.Module):

    def __init__(self, n_classes=20):
        super(Net, self).__init__()
        self.n_classes = n_classes
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, self.n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def feature_list(self, x):
        out_list = []
        x = self.stage1(x)
        x = self.stage2(x) # .detach()
        out_list.append(x)

        x = self.stage3(x)
        out_list.append(x)
        x = self.stage4(x)
        x = torchutils.gap2d(x, keepdims=True)
        out_list.append(x)

        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        return x, out_list


    def forward(self, x, return_feature=False):

        x = self.stage1(x)
        x = self.stage2(x) # .detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        feat = x
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)
        if return_feature:
            return x, feat
        else:
            return x

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self, n_classes=20):
        super(CAM, self).__init__(n_classes=n_classes)

    def forward(self, x, separate=False, get_feature=False, return_norelu=False, is_RIB=False, is_cls=False, cats=None):
        if is_RIB:
            if is_cls:
                x = self.stage1(x)
                x = self.stage2(x).detach()
                x = self.stage3(x)
                x = self.stage4(x)
                x = torchutils.gap2d(x, keepdims=True)
                x = self.classifier(x)
                x = x.view(-1, 20)
            else:
                x = self.stage1(x)
                x = self.stage2(x).detach()
                x = self.stage3(x)
                x = self.stage4(x)
                x = F.conv2d(x, self.classifier.weight)
                indss = None

            if cats is None:
                return x
            else:
                return x, indss
        else:
            out_list = []
            x = self.stage1(x)

            x = self.stage2(x)
            out_list.append(x)
            x = self.stage3(x)
            out_list.append(x)
            x = self.stage4(x)
            out_list.append(torchutils.gap2d(x, keepdims=True))

            x = F.conv2d(x, self.classifier.weight)
            if separate:
                if get_feature:
                    return x, out_list
                else:
                    return x

            if return_norelu:
                norelu = x

            x = F.relu(x)

            x = x[0] + x[1].flip(-1)
            if return_norelu:
                return x, norelu[0] + norelu[1].flip(-1)
            else:
                return x
