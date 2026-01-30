import torch.nn as nn
from MinkowskiEngine import (MinkowskiDepthwiseConvolution,
                             MinkowskiConvolution,
                             MinkowskiBatchNorm,
                             MinkowskiInstanceNorm,
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

    def __init__(self, in_planes, planes, reps, strides=1, start_with_relu=True, exit_flow=False, D=2, norm_layer=None):
        super(SparseBlock, self).__init__()
        if norm_layer is None:
            raise ValueError("norm_layer should be provided")
        self.norm_layer = norm_layer
        if planes != in_planes or strides != 1:
            self.skip = nn.Sequential(MinkowskiConvolution(in_planes, planes, kernel_size=1,
                                                           stride=strides, dimension=D),
                                      self.norm_layer(planes))
        else:
            self.skip = None

        self.activation = MinkowskiReLU()
        rep = []

        for i in range(reps):
            if start_with_relu or i != 0:
                rep.append(self.activation)
            if exit_flow and i == 0:
                rep.append(SparseSeparableConv(in_planes, in_planes, kernel_size=3, D=D))
                rep.append(self.norm_layer(in_planes))
            else:
                rep.append(SparseSeparableConv(in_planes, planes, kernel_size=3, D=D))
                rep.append(self.norm_layer(planes))
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

    def __init__(self, D=2, norm_layer="batch"):
        super(SparseXception, self).__init__()
        if norm_layer == "batch":
            norm_layer_1 = MinkowskiBatchNorm
            norm_layer_2 = MinkowskiBatchNorm
            norm_layer_3 = MinkowskiBatchNorm
        elif norm_layer == "instance":
            norm_layer_1 = MinkowskiInstanceNorm
            norm_layer_2 = MinkowskiInstanceNorm
            norm_layer_3 = MinkowskiInstanceNorm
        elif norm_layer == "hybrid":
            norm_layer_1 = MinkowskiBatchNorm
            norm_layer_2 = MinkowskiInstanceNorm
            norm_layer_3 = MinkowskiInstanceNorm
        elif norm_layer == "hybrid_only_end":
            norm_layer_1 = MinkowskiBatchNorm
            norm_layer_2 = MinkowskiBatchNorm
            norm_layer_3 = MinkowskiInstanceNorm
        else:
            raise ValueError("Unknown norm layer: {}".format(norm_layer))

        self.entry_flow = nn.Sequential(SparseBlock(64, 128, 2, 2, start_with_relu=False,
                                                    D=D, norm_layer=norm_layer_1),
                                        SparseBlock(128, 256, 2, 2,
                                                    D=D, norm_layer=norm_layer_1),
                                        SparseBlock(256, 728, 2, 2,
                                                    D=D, norm_layer=norm_layer_1))

        self.middle_flow = nn.Sequential(*[SparseBlock(728, 728, 3, 1, D=D,
                                                       norm_layer=norm_layer_2) for _ in range(8)])

        self.exit_flow = nn.Sequential(SparseBlock(728, 1024, 2, 2,
                                                   D=D, exit_flow=True, norm_layer=norm_layer_3),
                                       SparseSeparableConv(1024, 1536, 3, D=D),
                                       norm_layer_3(1536),
                                       MinkowskiReLU(),
                                       SparseSeparableConv(1536, 2048, 3, D=D),
                                       norm_layer_3(2048),
                                       MinkowskiReLU(),
                                       )

    def forward(self, x):
        x1 = self.entry_flow(x)
        x2 = self.middle_flow(x1)
        x3 = self.exit_flow(x2)

        return x3


def sparsexception(**kwargs):
    return SparseXception(**kwargs)
