import torch
import torch.nn as nn
import torch.nn.functional as nnf
import natten
# from .MyNAT3D import NeighborhoodAttention3D
from natten import NeighborhoodAttention3D

class FConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            # nn.InstanceNorm3d(out_channels),

            nn.GELU(),

        )
        # self.res = nn.Sequential(
        #     nn.Conv3d(out_channels,out_channels, 3,1,1),nn.GELU()
        # )

        # self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        # # self.norm = nn.InstanceNorm3d(out_channels)
        # self.activation =

    def forward(self, x):
        x = self.layers(x)
        # x = self.res(x) + x
        # x = self.res(x) + x
        # out = self.norm(out)
        # out = self.activation(out)
        return x


class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm3d(out_channels),

            nn.GELU(),

        )
        self.res = nn.Sequential(
            nn.Conv3d(out_channels,out_channels, 3,1,1),nn.GELU()
        )
        # self.res2 = nn.Sequential(
        #     nn.Conv3d(out_channels,out_channels, 3,1,1),nn.GELU()
        # )
        # self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        # # self.norm = nn.InstanceNorm3d(out_channels)
        # self.activation =

    def forward(self, x):
        x = self.layers(x)
        x = self.res(x) + x
        # x = self.res2(x) + x
        # out = self.norm(out)
        # out = self.activation(out)
        return x

class NaInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, channels, heads,kernal_size=3, alpha=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(channels)
        self.main = NeighborhoodAttention3D(channels, heads, kernal_size, rel_pos_bias=False)
        # self.norm = nn.InstanceNorm3d(channels)
        # self.activation = nn.GELU()
        # self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        feat = x
        x = self.ln(x.permute(0, 2,3,4, 1))
        x = self.main(x)

        # x = self.norm(x)
        # x = self.activation(x)
        x = x.permute(0,4, 1,2,3)+ feat

        # x = self.norm(x)
        # x = self.activation(x)
        return x
class CoNaTEncoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=8, heads=(1,2,4,4)):
        super(CoNaTEncoder, self).__init__()

        c = first_out_channel

        self.conv0 = nn.Sequential(
            ConvInsBlock(in_channel, c),
            #NaInsBlock(c, heads[0])
        )

        self.conv1 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock( c, 2 * c),
            NaInsBlock(2 * c, heads[1])
        )

        self.conv2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(2 * c, 4 * c),
            NaInsBlock(4 * c, heads[2])
        )

        self.conv3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(4 * c, 4* c),
            NaInsBlock(4 * c, heads[3])
        )

        # self.conv4 = nn.Sequential(
        #     nn.AvgPool3d(2),
        #     ConvInsBlock(8 * c, 16 * c),
        #     NaInsBlock(16 * c, heads[4])
        # )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        # out4 = self.conv4(out3)  # 1/8

        return out0, out1, out2, out3#, out4

if __name__ == '__main__':
    A = torch.randn(1,1,160,192,224).cuda()
    model = CoNaTEncoder().cuda()
    out = model(A)
    for o in out:
        print(o.shape)

    # model = NeighborhoodAttention3D(32, 4, 3).cuda()
    # input = torch.randn(1, 16,16,16, 32).cuda()
    # out = model(input)
    # print(out.shape)