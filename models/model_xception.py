import torch.nn as nn
from MinkowskiEngine import (MinkowskiDepthwiseConvolution,
                             MinkowskiConvolution,
                             MinkowskiBatchNorm,
                             MinkowskiReLU,
                             MinkowskiMaxPooling,
                             SparseTensor,
                             MinkowskiAvgPooling)


class SparseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, bias=False, D=2):
        super(SparseSeparableConv, self).__init__()

        self.depthwise_conv = MinkowskiDepthwiseConvolution(in_channels, kernel_size=kernel_size, stride=stride,
                                                            dilation=dilation, bias=bias, dimension=D)
        self.conv_1x1 = MinkowskiConvolution(in_channels, out_channels, kernel_size=1, bias=bias, dimension=D)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.conv_1x1(x)
        return x


class SparseBlock(nn.Module):

    def __init__(self, in_planes, planes, reps, strides=1, start_with_relu=True, exit_flow=False, D=2):
        super(SparseBlock, self).__init__()
        if planes != in_planes or strides != 1:
            self.skip = nn.Sequential(MinkowskiConvolution(in_planes, planes, kernel_size=1,
                                                           stride=strides, dimension=D),
                                      MinkowskiBatchNorm(planes))
        else:
            self.skip = None

        self.activation = MinkowskiReLU()
        rep = []

        for i in range(reps):
            if start_with_relu or i != 0:
                rep.append(self.activation)
            if exit_flow and i == 0:
                rep.append(SparseSeparableConv(in_planes, in_planes, kernel_size=3, D=D))
                rep.append(MinkowskiBatchNorm(in_planes))
            else:
                rep.append(SparseSeparableConv(in_planes, planes, kernel_size=3, D=D))
                rep.append(MinkowskiBatchNorm(planes))
                in_planes = planes

        if strides != 1:
            rep.append(MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D))

        self.model = nn.Sequential(*rep)

    def forward(self, x):
        if self.skip:
            x_skip = self.skip(x)
        else:
            x_skip = x
        return self.model(x) + x_skip


class SparseXception(nn.Module):

    def __init__(self, D=2, normalized=False):
        super(SparseXception, self).__init__()

        self.entry_flow = nn.Sequential(SparseBlock(64, 128, 2, 2, start_with_relu=False, D=D),
                                        SparseBlock(128, 256, 2, 2, D=D),
                                        SparseBlock(256, 728, 2, 2, D=D))
        self.middle_flow = nn.Sequential(*[SparseBlock(728, 728, 3, 1, D=D) for _ in range(8)])

        self.exit_flow = nn.Sequential(SparseBlock(728, 1024, 2, 2, D=D, exit_flow=True),
                                       SparseSeparableConv(1024, 1536, 3, D=D),
                                       MinkowskiBatchNorm(1536),
                                       MinkowskiReLU(),
                                       SparseSeparableConv(1536, 2048, 3, D=D),
                                       MinkowskiBatchNorm(2048),
                                       MinkowskiReLU(),
                                       )

    def forward(self, x):
        x1 = self.entry_flow(x)
        x2 = self.middle_flow(x1)
        x3 = self.exit_flow(x2)

        return x3


def sparsexception(**kwargs):
    return SparseXception(**kwargs)
