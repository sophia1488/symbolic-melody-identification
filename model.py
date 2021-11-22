import torch
import torch.nn as nn


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_mean.data.fill_(0.0)
    bn.running_var.data.fill_(1.0)


class CNN_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.bn = nn.BatchNorm2d(21)
        self.conv1_layer = nn.Conv2d(1, 21, (32,16))
        self.conv2_layer = nn.Conv2d(21, 441, (32,16))
        self.deconv1_layer = nn.ConvTranspose2d(441, 21, (32,16))
        self.deconv2_layer = nn.ConvTranspose2d(21, 1, (32,16))

        # in_channel, out_channel, kernel_size
        self.conv1 = nn.Sequential(
            self.conv1_layer,
            self.bn,
            nn.Dropout(0.3),
        )
        self.conv2 = nn.Sequential(
            self.conv2_layer,
            nn.ReLU()
        )
        self.deconv1 = nn.Sequential(
            self.deconv1_layer,
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.deconv2 = nn.Sequential(
            self.deconv2_layer,
            nn.Sigmoid()
        )

    def init_weights(self):
        init_bn(self.bn)
        init_layer(self.conv1_layer)
        init_layer(self.conv2_layer)
        init_layer(self.deconv1_layer)
        init_layer(self.deconv2_layer)

    def forward(self, x):
        # (batch, 1, 128, 64)
        x = self.conv1(x)
        # (batch, 21, 97, 49)
        x = self.conv2(x)
        # (batch, 441, 66, 34)
        x = self.deconv1(x)
        # (batch, 21, 97, 49)
        x = self.deconv2(x)
        # (batch, 1, 128, 64)
        return x
