import torch
import torch.nn as nn


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class HFFB(nn.Module):
    def __init__(self, nc=64):
        super(HFFB, self).__init__()
        self.c1 = conv_layer(nc, nc, 3, 1, 1)
        self.d1 = conv_layer(nc, nc // 2, 3, 1, 1)  # rate=1
        self.d2 = conv_layer(nc, nc // 2, 3, 1, 2)  # rate=2
        self.d3 = conv_layer(nc, nc // 2, 3, 1, 3)  # rate=3
        self.d4 = conv_layer(nc, nc // 2, 3, 1, 4)  # rate=4
        self.d5 = conv_layer(nc, nc // 2, 3, 1, 5)  # rate=5
        self.d6 = conv_layer(nc, nc // 2, 3, 1, 6)  # rate=6
        self.d7 = conv_layer(nc, nc // 2, 3, 1, 7)  # rate=7
        self.d8 = conv_layer(nc, nc // 2, 3, 1, 8)  # rate=8
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.c2 = conv_layer(nc * 4, nc, 1, 1, 1)  # 256-->64

    def forward(self, input):
        output1 = self.act(self.c1(input))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)
        d5 = self.d5(output1)
        d6 = self.d6(output1)
        d7 = self.d7(output1)
        d8 = self.d8(output1)

        add1 = d1 + d2
        add2 = add1 + d3
        add3 = add2 + d4
        add4 = add3 + d5
        add5 = add4 + d6
        add6 = add5 + d7
        add7 = add6 + d8

        combine = torch.cat([d1, add1, add2, add3, add4, add5, add6, add7], 1)
        output2 = self.c2(self.act(combine))
        output = input + output2.mul(0.2)

        return output
