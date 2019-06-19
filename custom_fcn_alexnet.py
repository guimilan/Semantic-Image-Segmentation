import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.modules.loss


# custom_fcn_alexnet
class CustomFCNAlexnet(nn.Module):
    # Alexnet-based FCN implementation
    # for the sake of simplicity, at first the input will have an assumed dimension
    # of 227x227x3
    def __init__(self, num_classes=90, alexnet=None):
        super(CustomFCNAlexnet, self).__init__()

        # Convolution layers for feature extraction
        self.conv1 = nn.Sequential(
            alexnet.features[0],  # conv2d(227x227x3, 55x55x64, kernel=11x11, stride=4, padding=2)
            alexnet.features[1],  # relu
            alexnet.features[2],  # maxpool2d(55x55x64, 27x27x64, kernel 3x3, stride=2, padding=0) pool 1
        )
        self.conv2 = nn.Sequential(
            alexnet.features[3],  # conv2d(27x27x64, 27x27x192, kernel=5x5, stride=1, padding=1)
            alexnet.features[4],  # relu
            alexnet.features[5],  # maxpool2d(27x27x192, 13x13x192, kernel=3x3, stride=2, padding=0) pool 2
        )
        self.conv3 = nn.Sequential(
            alexnet.features[6],  # conv2d(13x13x192, 13x13x384, kernel=3x3, stride=1, padding=1)
            alexnet.features[7],  # relu
        )
        self.conv4 = nn.Sequential(
            alexnet.features[8],  # conv2d(13x13x384, 13x13x256, kernel=3x3, stride=1, padding=1)
            alexnet.features[9]  # relu
        )
        self.conv5 = nn.Sequential(
            alexnet.features[10],  # conv2d(13x13x256, 13x13x256, kernel=3x3, stride=1, padding=1)
            alexnet.features[11],  # relu
        )
        self.conv6 = alexnet.features[12]  # maxpool2d(13x13x256, 6x6x256, kernel=3x3, stride=2) pool 3

        # size-1 convolution for pixel-by-pixel prediction
        self.score_conv = nn.Conv2d(256, num_classes, 1)

        # Deconvolution layers for restoring the original image
        self.deconv1 = nn.ConvTranspose2d(num_classes, 256, kernel_size=3, stride=2)

        self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2)

        self.deconv3 = nn.ConvTranspose2d(64, 192, kernel_size=3, stride=2)

        self.deconv4 = nn.ConvTranspose2d(192, num_classes, kernel_size=8, stride=4)

    #Forward passes the data
    #Skip connections are formed by summing together the two connected layers' output
    def forward(self, x):
        #print('input shape', x.size())
        out_conv1 = self.conv1(x)
        # print('conv 1 output shape', out_conv1.size())

        out_conv2 = self.conv2(out_conv1)
        # print('conv 2 output shape', out_conv2.size())

        out_conv3 = self.conv3(out_conv2)
        # print('conv 3 output shape', out_conv3.size())

        out_conv4 = self.conv4(out_conv3)
        # print('conv 4 output shape', out_conv4.size())

        out_conv5 = self.conv5(out_conv4)
        # print('conv 5 output shape', out_conv5.size())

        out_conv6 = self.conv6(out_conv5)
        # print('conv 6 output shape', out_conv6.size())

        out_score_conv = self.score_conv(out_conv6)
        # print('score conv output shape', out_score_conv.size())

        out_deconv1 = self.deconv1(out_score_conv)
        # print('deconv1 output shape', out_deconv1.size())

        out_deconv2 = self.deconv2(out_deconv1 + out_conv5)
        # print('deconv 2 output shape', out_deconv2.size())

        out_deconv3 = self.deconv3(out_deconv2 + out_conv1)
        # print('deconv 3 output shape', out_deconv3.size())

        out_deconv4 = self.deconv4(out_deconv3)

        return out_deconv4
