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
		self.conv1 = nn.Sequential(
			nn.Conv2d(input=num_classes, out_channels=96, kernel_size=11, stride=4, padding=100),
			nn.relu(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(input=96, out_channels=256, kernel_size=5, padding=2, groups=2),
			nn.relu(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75),
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(kernel_size=3, input=256, out_channels=384, padding=1),
			nn.relu(),
		)

		self.conv4 = nn.Sequential(
			nn.Conv2d(),
			nn.relu(),
		)

		self.conv5 = nn.Sequential(
			nn.Conv2d(),
			nn.relu(),
			nn.pool(),
		)

		self.conv6 = nn.Sequential(
			nn.Conv2d(),
			nn.relu(),
			torch.nn.Dropout(),
		)

		self.conv7 = nn.Sequential(
			nn.Conv2d(),
			nn.relu(),
			torch.nn.Dropout(),	
		)

		self.conv8 = nn.Conv2d()
		self.deconv = nn.ConvTranspose2d()
		self.softmax = nn.Softmax(1)

	def forward(self, x)

