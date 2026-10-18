# Adapted from OpenOOD v1.5 (https://github.com/Jingkang50/OpenOOD),
# openood/networks/resnet18_32x32.py and openood/networks/resnet18_224x224.py,
# as redistributed in the OODD (CVPR 2025) code release.
# Copyright (c) 2021 Jingkang Yang and the OpenOOD contributors. MIT License:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Changes: chỉ giữ phần dùng để trích đặc trưng (bỏ Bottleneck, forward_threshold, intermediate_forward, get_fc);
# tên tầng giữ nguyên để nạp được checkpoint chính thức.
"""Mạng ResNet-18 của OpenOOD v1.5 (bản 32x32 cho CIFAR, 224x224 cho ImageNet-200)."""
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import BasicBlock as TorchvisionBasicBlock
from torchvision.models.resnet import ResNet


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18_32x32(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=(2, 2, 2, 2), num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        feature = self.avgpool(out).view(out.size(0), -1)
        logits = self.fc(feature)
        return (logits, feature) if return_feature else logits


class ResNet18_224x224(ResNet):
    def __init__(self, block=TorchvisionBasicBlock, layers=(2, 2, 2, 2), num_classes=1000):
        super().__init__(block=block, layers=list(layers), num_classes=num_classes)

    def forward(self, x, return_feature=False):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        feature = self.avgpool(out).view(out.size(0), -1)
        logits = self.fc(feature)
        return (logits, feature) if return_feature else logits
