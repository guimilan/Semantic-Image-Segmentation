import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
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

def load_dataset(batch_size):
	transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])	
								 
	train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
	
	test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	return train_loader, test_loader, classes

class CNN(nn.Module):
	#Implementacao da AlexNet, baseada na implementacao do proprio pytorch
	def __init__(self, num_classes=1000, alexnet=None):
		super(CNN, self).__init__()
		if(alexnet is not None):
			self.features = alexnet.features
			self.avgpool = alexnet.avgpool
			self.classifier = alexnet.classifier
		else:
			self.features = nn.Sequential(
				nn.Conv2d(3,64, kernel_size=11, stride=4, padding=2),#requires_grad=True por padrao
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=3, stride=2),

				nn.Conv2d(64, 192, kernel_size=5, padding=2),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=3, stride=2),

				nn.Conv2d(192, 384, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),

				nn.Conv2d(384, 256, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),

				nn.Conv2d(256, 256, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=3, stride=2),
			)
			self.avg_pool = nn.AdaptiveAvgPool2d((6,6))

			self.classifier = nn.Sequential(
				nn.Dropout(),
				nn.Linear(256*6*6, 4096),
				nn.ReLU(inplace=True),
				
				nn.Dropout(),
				nn.Linear(4096, 4096),
				nn.ReLU(inplace=True),

				nn.Linear(4096, num_classes),
			)


	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1) #-1 infere o tamanho
		x = self.classifier(x)
		return x

def fit(model, train_dataset, device):
	criterion = nn.CrossEntropyLoss()#Funcao de erro: entropia cruzada
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	#Gradiente descendente estocastico
	#Taxa de aprendizado inicial: 0.001
	#Aceleracao: 0.9

	for epoch in range(2):
		running_loss = 0.0
		model.train()#Coloca flag de treinamento para direcionar o comportamento correto
		#das camadas de dropout e batch normalization

		for index, data in enumerate(train_dataset, 0):
			#data contem um batch de batch_size samples do dataset e suas classes
			samples, labels = data
			samples, labels = samples.to(device), labels.to(device)
			#samples = samples.to(self.device)
			#labels = labels.to(self.device)

			optimizer.zero_grad()#Zera o gradiente (se nao fizer isso, ele acumula e aplica o acumulado
			#na hora de atualizar os pesos)
			output = model(samples)#Pega a saida da rede (faz forward pass. nao se chama o 
			#forward pass diretamente em pytorch)
			loss = criterion(output, labels)#Calcula o erro
			loss.backward()#Faz backpropagation e calcula o quanto cada parametro deve ser otimizado
			optimizer.step()#Atualiza cada parametro com os valores calculados no backpropagation

			running_loss += loss.item()#?
			if index % 100 == 99:
				print('[%d %5d] loss %.3f' % (epoch+1, index+1, running_loss/2000))
				running_loss = 0.0

	print('finished training')

def validate(model, test_dataset, device):
	correct = 0
	total = 0
	model.eval()#Coloca a flag de validacao para direcionar o comportamento correto
	#das camadas de dropout e batch normalization
	with torch.no_grad():
		for data in test_dataset:
			samples, labels = data
			samples = samples.to(device)
			labels = labels.to(device)
			outputs = model.forward(samples)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	print('Accuracy of the network: %d %%' % (100 * correct/total))
	return correct, total

def main():
	print("Cuda availability status:", torch.cuda.is_available())
	print('setting device...')
	torch.cuda.set_device(0)
	print('device set')
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	print("loading dataset")
	train, test, classes = load_dataset(16)

	print("dataset loaded. instantiating alexnet...")
	alexnet = models.alexnet(pretrained=True)
	print('alexnet instantiated. instantiating model...')
	cnn = CNN(10, alexnet)
	print('done. sending model to gpu')
	cnn.cuda()
	print('sending successful. training...')
	print('training')
	fit(cnn, train, device)
	#print('validating')
	#validate(cnn, test, device)


	#validate(alexnet, test, device)
	#print('validated')'''
	return 0

if __name__ == '__main__':
	main()