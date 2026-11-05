"""Mạng CIFAR 32x32 cho vòng 61 (đánh đổi độ trễ - chất lượng theo backbone), cùng giao diện ResNet18_32x32.

ResNet-20: He et al., CVPR 2016, mục 4.2 (3 tầng x 3 khối, 16/32/64 kênh).
MobileNetV2: Sandler et al., CVPR 2018; bản CIFAR theo kuangliu/pytorch-cifar (MIT): conv đầu và tầng thứ hai stride 1.
ResNet-50: khối Bottleneck (3, 4, 6, 3) với thân 3x3 stride 1, ko maxpool, như bản 32x32 của ResNet-18 OpenOOD.
ResNet-18 nhúng: ResNet18_32x32 thêm tầng tuyến tính 512 → d trước fc; đặc trưng là vector d chiều được học.
"""
import torch.nn.functional as F
from torch import nn

from .openood_nets import BasicBlock, ResNet18_32x32


class _Head(nn.Module):
    """forward(x, return_feature) -> logits | (logits, feature), như mạng OpenOOD."""

    def head(self, feature, return_feature):
        logits = self.fc(feature)
        return (logits, feature) if return_feature else logits


class ResNet20_32x32(_Head):
    def __init__(self, num_classes=10, n=3):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, n, 1)
        self.layer2 = self._make_layer(32, n, 2)
        self.layer3 = self._make_layer(64, n, 2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))  # tắt 1x1 khi đổi chiều (phương án B của bài)
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        out = self.layer3(self.layer2(self.layer1(F.relu(self.bn1(self.conv1(x))))))
        return self.head(F.adaptive_avg_pool2d(out, 1).flatten(1), return_feature)


class _InvertedResidual(nn.Module):
    def __init__(self, cin, cout, expansion, stride):
        super().__init__()
        mid = expansion * cin
        self.stride = stride
        self.conv1 = nn.Conv2d(cin, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, cout, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride == 1 and cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out + self.shortcut(x) if self.stride == 1 else out


class MobileNetV2_32x32(_Head):
    # (expansion, out_planes, num_blocks, stride); stride 2 → 1 ở tầng thứ hai cho ảnh 32x32
    CFG = ((1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1))

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        layers, cin = [], 32
        for e, cout, n, s in self.CFG:
            for st in [s] + [1] * (n - 1):
                layers.append(_InvertedResidual(cin, cout, e, st))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x, return_feature=False):
        out = self.layers(F.relu(self.bn1(self.conv1(x))))
        out = F.relu(self.bn2(self.conv2(out)))
        return self.head(F.adaptive_avg_pool2d(out, 1).flatten(1), return_feature)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, 4 * planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != 4 * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, 4 * planes, 1, stride, bias=False),
                                          nn.BatchNorm2d(4 * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return F.relu(self.bn3(self.conv3(out)) + self.shortcut(x))


class ResNet50_32x32(ResNet18_32x32):
    def __init__(self, num_classes=10):
        super().__init__(block=_Bottleneck, num_blocks=(3, 4, 6, 3), num_classes=num_classes)


class ResNet18Embed_32x32(_Head):
    """ResNet-18 + Linear(512 → dim) + BN; vector nhúng là đặc trưng, fc phân loại trên nó."""

    def __init__(self, num_classes=10, dim=128):
        super().__init__()
        self.body = ResNet18_32x32(num_classes=num_classes)
        self.body.fc = nn.Identity()
        self.embed = nn.Sequential(nn.Linear(512, dim), nn.BatchNorm1d(dim))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x, return_feature=False):
        _, f = self.body(x, return_feature=True)
        return self.head(self.embed(f), return_feature)


class ConvNeXtTiny_32x32(_Head):
    """ConvNeXt-Tiny (CVPR 2022) cho ảnh 32x32; vector đặc trưng 768 chiều."""

    def __init__(self, num_classes=10):
        super().__init__()
        import torchvision.models as models
        m = models.convnext_tiny(num_classes=num_classes)
        self.features = m.features
        self.avgpool = m.avgpool
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x, return_feature=False):
        f = self.features(x)
        f = self.avgpool(f).flatten(1)
        return self.head(f, return_feature)


CIFAR_NETS = {"resnet20_32x32": ResNet20_32x32, "mobilenetv2_32x32": MobileNetV2_32x32,
              "resnet50_32x32": ResNet50_32x32, "resnet18embed_32x32": ResNet18Embed_32x32,
              "convnext_tiny_32x32": ConvNeXtTiny_32x32}
