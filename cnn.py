import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

def load_dataset():
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
	
	train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
	
	test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	return train_loader, test_loader, classes

class CNN(nn.Module):
	#Implementacao da AlexNet, baseada na implementacao do proprio pytorch
	def __init__(self, device):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(3,6,5)
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(6,16,5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
		self.device = device
		self.to(device)
		
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x))) #Aplica relu e faz pooling sobre a saida da 1a conv
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def fit(self, train_dataset):
		criterion = nn.CrossEntropyLoss()#Funcao de erro: entropia cruzada
		optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
		#Gradiente descendente estocastico
		#Taxa de aprendizado inicial: 0.001
		#Aceleracao: 0.9

		for epoch in range(2):
			running_loss = 0.0

			for index, data in enumerate(train_dataset, 0):
				#data contem um batch de 4 samples do dataset e suas classes
				samples, labels = data
				samples, labels = samples.to(self.device), labels.to(self.device)
				samples = samples.to(self.device)
				labels = labels.to(self.device)

				optimizer.zero_grad()#Zera o gradiente (se nao fizer isso, ele acumula e aplica o acumulado
				#na hora de atualizar os pesos)
				predicted = self(samples)#Pega a saida da rede
				loss = criterion(predicted, labels)#Calcula o erro
				loss.backward()#Faz backpropagation e calcula o quanto cada parametro deve ser otimizado
				optimizer.step()#Atualiza cada parametro com os valores calculados no backpropagation

				running_loss += loss.item()#?
				if index % 2000 == 1999:
					print('[%d %5d] loss %.3f' % (epoch+1, index+1, running_loss/2000))
					running_loss = 0.0

		print('finished training')

	def validate(self, test_dataset):
		correct = 0
		total = 0
		with torch.no_grad():
			for data in test_dataset:
				samples, labels = data
				samples = samples.to(self.device)
				labels = labels.to(self.device)
				outputs = self.forward(samples)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		print('Accuracy of the network: %d %%' % (100 * correct/total))
		return correct, total

def main():
	print("Cuda availability status:", torch.cuda.is_available())
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cnn = CNN(device)
	
	train, test, classes = load_dataset()
	print('train', train)
	cnn.fit(train)
	cnn.validate(test)

	return 0

if __name__ == '__main__':
	main()