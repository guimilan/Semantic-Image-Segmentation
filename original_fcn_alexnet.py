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
			nn.Conv2d(),
			nn.relu(),
			nn.pool(),
			norm,
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(),
			nn.relu(),
			nn.pool(),
			norm,
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(),
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
			dropout,
		)

		self.conv7 = nn.Sequential(
			nn.Conv2d(),
			nn.relu(),
			dropout,	
		)

		self.conv8 = nn.Conv2d()
		self.deconv = nn.ConvTranspose2d()



