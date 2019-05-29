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

def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

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
	#Implementacao da AlexNet, baseada na implementacao do proprio pytorch
	def __init__(self, num_classes=1000, alexnet=None):
		super(CNN, self).__init__()
		#Quebrando a alexnet nos fragmentos que modificam a dimensao da entrada
		#Isto eh feito para permitir pegar a saida de cada fragmento durante o forward pass,
		#podendo, entao, fazer a skip connection das camadas no inicio da rede com as camadas no final
		#percententes a mesma faixa de tamanho

		#faixas de tamanho: 227x227, 55x55, 27x27, 13x13 e 6x6
		self.conv1 = nn.Sequential(
			alexnet.features[0], #conv2d(227x227x3, 55x55x64, kernel=11x11, stride=4, padding=2)
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
		
		self.score_conv = nn.Conv2d(256, num_classes, 1)
		'''if(alexnet is not None):
			self.alexnet = alexnet.features
			for param in self.alexnet.parameters():
				param.requires_grad = False
		else:
			self.alexnet = nn.Sequential(
				#requires_grad=True por padrao
				nn.Conv2d(3,64, kernel_size=11, stride=4, padding=2),#dim da saida: 55x55x64
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=3, stride=2),#dim da saida: 27x27x64

				nn.Conv2d(64, 192, kernel_size=5, padding=2),#dim da saida: 27x27x192
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=3, stride=2),#dim da saida: 13x13x192

				nn.Conv2d(192, 384, kernel_size=3, padding=1),#dim da saida: 13x13x384
				nn.ReLU(inplace=True),

				nn.Conv2d(384, 256, kernel_size=3, padding=1),#dim da saida: 13x13x256
				nn.ReLU(inplace=True),

				nn.Conv2d(256, 256, kernel_size=3, padding=1),#dim da saida: 13x13x256
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=3, stride=2),#dim da saida: 6x6x256
			)'''

		#entrada: 6x6x256, saida: 13x13x256
		self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2)
		#entrada: 13x13x256 (ligar com a saida da conv5), saida: 27x27x64
		self.deconv2 = nn.ConvTranspose2d(384, 64, kernel_size=3, stride=2)
		#entrada: 27x27x64 (ligar com a saida da first_conv), saida: 55x55x64
		self.deconv3 = nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2)
		#entrada: 55x55x64 (ligar com a saida da input_conv (renomear pra first_conv, corrigir nome das
		# outras camadas. first_conv atual vira second_conv e assim por diante)), saida: 227x227xnum_classes
		self.deconv4 = nn.ConvTranspose2d(192, num_classes, kernel_size=11, stride=4)
		#Checar se precisar de uma convolucao que nao altera h, w e d na camada de saida

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)
		x = self.conv7(x)
		x = self.score_conv(x)
		x = self.deconv1(x)
		x = self.deconv2(x)
		x = self.deconv3(x)
		x = self.deconv4(x)
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
			samples, labels = samples.to(device), labels.to(device)
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
	train, test, classes = load_cifar10(16)

	print("dataset loaded. instantiating alexnet...")
	alexnet = models.alexnet(pretrained=True)
	print('alexnet instantiated. instantiating model...')
	cnn = CNN(len(classes), alexnet)
	print('done. sending model to gpu')
	cnn.cuda()
	print('sending successful. training...')
	print('training')
	fit(cnn, train, device)
	print('validating')
	validate(cnn, test, device)

	#validate(alexnet, test, device)
	#print('validated')'''
	return 0

if __name__ == '__main__':
	main()