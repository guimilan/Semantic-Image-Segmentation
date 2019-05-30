import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import utils

#plots an image
def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

#(legacy code)Loads the cifar10 dataset for image classification
#this was used to test fine-tuning a pretrained alexnet, and also
#to learn the tools for data loading in pytorch
def load_cifar10(batch_size):
	resize = transforms.Resize((224, 224))
	toTensor = transforms.ToTensor()
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	transform = transforms.Compose([resize, toTensor, normalize])	

	train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
	
	test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	return train_loader, test_loader, classes

class CNN(nn.Module):
	#Alexnet-based FCN implementation
	#for the sake of simplicity, at first the input will have an assumed dimension
	#of 227x227x3
	def __init__(self, num_classes=1000, alexnet=None):
		super(CNN, self).__init__()

		#Convolution layers for feature extraction
		#Sizes: 227x227, 55x55, 27x27, 13x13 e 6x6
		self.conv1 = nn.Sequential(
			alexnet.features[0],#conv2d(227x227x3, 55x55x64, kernel=11x11, stride=4, padding=2)
			alexnet.features[1],#relu
			alexnet.features[2],#maxpool2d(55x55x64, 27x27x64, kernel 3x3, stride=2, padding=0) pool 1
		)
		self.conv2 = nn.Sequential (
			alexnet.features[3],#conv2d(27x27x64, 27x27x192, kernel=5x5, stride=1, padding=1)
			alexnet.features[4],#relu
			alexnet.features[5],#maxpool2d(27x27x192, 13x13x192, kernel=3x3, stride=2, padding=0) pool 2
		)
		self.conv3 = nn.Sequential (
			alexnet.features[6],#conv2d(13x13x192, 13x13x384, kernel=3x3, stride=1, padding=1)
			alexnet.features[7],#relu
		)
		self.conv4 = nn.Sequential (
			alexnet.features[8],#conv2d(13x13x384, 13x13x256, kernel=3x3, stride=1, padding=1)
			alexnet.features[9]#relu
		)
		self.conv5 = nn.Sequential (
			alexnet.features[10],#conv2d(13x13x256, 13x13x256, kernel=3x3, stride=1, padding=1)
			alexnet.features[11],#relu
		)
		self.conv6 = alexnet.features[12]#maxpool2d(13x13x256, 6x6x256, kernel=3x3, stride=2) pool 3
		for param in self.parameters():
			param.required_grad = False

		#size-1 convolution for pixel-by-pixel prediction
		self.score_conv = nn.Conv2d(256, num_classes, 1)

		#Deconvolution layers for restoring the original image
		#input: 6x6x256, output: 13x13x256
		self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2)
		
		#input: 13x13x256 (skip-connect to conv5's output), output: 27x27x64
		self.deconv2 = nn.ConvTranspose2d(384, 64, kernel_size=3, stride=2)
		
		#input: 27x27x64 (skip-connect to conv1's output), output: 55x55x64
		self.deconv3 = nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2)
		
		#input: 55x55x64, output: 227x227xnum_classes
		self.deconv4 = nn.ConvTranspose2d(192, num_classes, kernel_size=11, stride=4)

	def forward(self, x):
		#Forward passes the data
		#Skip connections are formed by summing together the two connected layers' output 
		out_conv1 = self.conv1(x)
		out_conv2 = self.conv2(out_conv1)
		out_conv3 = self.conv3(out_conv2)
		out_conv4 = self.conv4(out_conv3)
		out_conv5 = self.conv5(out_conv4)
		out_conv6 = self.conv6(out_conv5)
		out_score_conv = self.score_conv(out_conv6)
		out_deconv1 = self.deconv1(out_score_conv)
		out_deconv2 = self.deconv2(out_deconv1+out_conv5)
		out_deconv3 = self.deconv3(out_deconv2+out_conv1)
		out_deconv4 = self.deconv4(out_deconv3)
		return out_deconv4

def fit(model, train_dataset, device):
	criterion = nn.CrossEntropyLoss()#Error function: cross entropy
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	#Stochastic gradient descent
	#Initial learning rate: 0.001
	#momentum: 0.9

	for epoch in range(2):
		running_loss = 0.0
		model.train()#Sets a flag indicating the code that follows performs training 
		#this makes sure the dropout and batch normalization layers perform as expected
		

		for index, data in enumerate(train_dataset, 0):
			#the variable data contains an entire batch of inputs and their associated labels 
			samples, labels = data
			samples, labels = samples.to(device), labels.to(device)#Sends the data to the GPU

			optimizer.zero_grad()#Zeroes the gradient, otherwise it will accumulate at every iteration
			#the result would be that the network would start taking huge parameter jumps as training went on 

			output = model(samples)#Forward passes the input data
			loss = criterion(output, labels)#Computes the error
			loss.backward()#Computes the gradient, yielding how much each parameter must be updated
			optimizer.step()#Updates each parameter according to the gradient

			running_loss += loss.item()
			if index % 100 == 99:
				print('[%d %5d] loss %.3f' % (epoch+1, index+1, running_loss/2000))
				running_loss = 0.0

	print('finished training')

def validate(model, test_dataset, device):
	correct = 0
	total = 0
	model.eval()#Sets the flag for evaluation
	#ensures batch normalization and dropout layers will stay inactive
	with torch.no_grad():
		for data in test_dataset:
			#loads a batch of data, sends it to GPU
			samples, labels = data
			samples, labels = samples.to(device), labels.to(device)
			
			#Forward passes the data
			outputs = model.forward(samples)
			
			#Takes the highest value from the output vector, computes the accuracy
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	print('Accuracy of the network: %d %%' % (100 * correct/total))
	return correct, total

#Main code for training (still using legacy code for alexnet finetuning and classification)
#Presently under modification
def main():
	#sets up cuda
	print("Cuda availability status:", torch.cuda.is_available())
	print('setting device...')
	torch.cuda.set_device(0)
	print('device set')
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	#loads cifar10 data
	print("loading dataset")
	train, test, classes = load_cifar10(16)

	#instantiates pretrained alexnet
	print("dataset loaded. instantiating alexnet...")
	alexnet = models.alexnet(pretrained=True)
	print('alexnet instantiated. instantiating model...')
	cnn = CNN(len(classes), alexnet)

	#Sends the network to the GPU
	print('done. sending model to gpu')
	cnn.cuda()
	print('sending successful. training...')
	
	#Performs training
	print('training')
	fit(cnn, train, device)
	
	#Performs validation
	print('validating')
	validate(cnn, test, device)

	#validate(alexnet, test, device)
	#print('validated')'''
	return 0

if __name__ == '__main__':
	main()