import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.modules.loss
import torch.nn.modules.normalization as norm

class OriginalFCNAlexnet(nn.Module):
	def __init__(self, num_classes=90):
		super(OriginalFCNAlexnet, self).__init__()
		#Adds one extra class to stand for the zero-padded pixels
		self.num_classes = num_classes + 1

		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=100),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75),
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, groups=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75),
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
			nn.ReLU(),
		)

		self.conv4 = nn.Sequential(
			nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=2),
			nn.ReLU(),
		)

		self.conv5 = nn.Sequential(
			nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, groups=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)

		self.conv6 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
			nn.ReLU(),
			torch.nn.Dropout(p=0.5, inplace=True),
		)

		self.conv7 = nn.Sequential(
			nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
			nn.ReLU(),
			torch.nn.Dropout(p=0.5, inplace=True),	
		)

		self.score_conv = nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=1, padding=0)
		self.deconv = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, 
			kernel_size=63, stride=32, bias=False)

	def forward(self, x):
		out1 = self.conv1(x)
		out2 = self.conv2(out1)
		out3 = self.conv3(out2)
		out4 = self.conv4(out3)
		out5 = self.conv5(out4)
		out6 = self.conv6(out5)
		out7 = self.conv7(out6)
		out_score = self.score_conv(out7)
		out_deconv = self.deconv(out_score)
		return out_deconv