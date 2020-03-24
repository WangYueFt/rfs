from __future__ import print_function

import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self, num_classes=-1):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64, momentum=1, affine=True, track_running_stats=False),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(64, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, is_feat=False):
        out = self.layer1(x)
        f0 = out
        out = self.layer2(out)
        f1 = out
        out = self.layer3(out)
        f2 = out
        out = self.layer4(out)
        f3 = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        feat = out

        if self.num_classes > 0:
            out = self.classifier(out)

        if is_feat:
            return [f0, f1, f2, f3, feat], out
        else:
            return out


def convnet4(**kwargs):
    """Four layer ConvNet
    """
    model = ConvNet(**kwargs)
    return model


if __name__ == '__main__':
    model = convnet4(num_classes=64)
    data = torch.randn(2, 3, 84, 84)
    feat, logit = model(data, is_feat=True)
    print(feat[-1].shape)
    print(logit.shape)
