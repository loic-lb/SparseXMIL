import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_planes, planes, reps, strides=1, start_with_relu=True, exit_flow=False):
        super(Block, self).__init__()
        if planes != in_planes or strides != 1:
            self.skip = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=strides, bias=False),
                                      nn.BatchNorm2d(planes))
        else:
            self.skip = None

        self.activation = nn.ReLU(inplace=True)
        rep = []

        for i in range(reps):
            if start_with_relu or i != 0:
                rep.append(self.activation)
            if exit_flow and i == 0:
                rep.append(SeparableConv2d(in_planes, in_planes, kernel_size=3, padding=1))
                rep.append(nn.BatchNorm2d(in_planes))
            else:
                rep.append(SeparableConv2d(in_planes, planes, kernel_size=3, padding=1))
                rep.append(nn.BatchNorm2d(planes))
                in_planes = planes

        if strides != 1:
            rep.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.model = nn.Sequential(*rep)

    def forward(self, x):
        out = self.model(x)
        if self.skip is not None:
            out += self.skip(x)
        else:
            out += x
        return out


class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()

        self.entry_flow = nn.Sequential(Block(64, 128, 2, strides=2, start_with_relu=False),
                                        Block(128, 256, 2, strides=2),
                                        Block(256, 728, 2, strides=2))
        self.middle_flow = nn.Sequential(*[Block(728, 728, 3) for _ in range(8)])
        self.exit_flow = nn.Sequential(Block(728, 1024, 2, strides=2, exit_flow=True),
                                       SeparableConv2d(1024, 1536, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(1536),
                                       nn.ReLU(inplace=True),
                                       SeparableConv2d(1536, 2048, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(2048),
                                       nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.entry_flow(x)
        x2 = self.middle_flow(x1)
        x3 = self.exit_flow(x2)

        return x3


def xception(**kwargs):
    return Xception(**kwargs)