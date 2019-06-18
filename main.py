import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.modules.loss
import matplotlib.pyplot as plt
import numpy as np
import utils
from pathlib import Path
import glob
from PIL import Image
import os
import imageio
import json
from pycocotools.coco import COCO
from timeit import default_timer as timer
from custom_fcn_alexnet import CustomFCNAlexnet
from coco_dataset import CocoDataset
import datetime


# plots an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# (legacy code)Loads the cifar10 dataset for image classification
# this was used to test fine-tuning a pretrained alexnet, and also
# to learn the tools for data loading in pytorch
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


def fit(model, train_dataset, device):
    criterion = nn.CrossEntropyLoss()  # Error function: cross entropy
    print('instantiating optimizer')
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print('optimizer instantiated')
    # Stochastic gradient descent
    # Initial learning rate: 0.001
    # momentum: 0.9

    epoch = 0
    running_loss = 1.0
    image_qtt = 0

    while epoch < 2 or running_loss < 10e-3:
        running_loss = 0.0
        print('epoch', epoch)
        model.train()  # Sets a flag indicating the code that follows performs training
        # this makes sure the dropout and batch normalization layers perform as expected
        print('loading new batch')
        batch_start = timer()
        for index, (samples, labels) in enumerate(train_dataset):
            batch_end  = timer()
            print('batch loaded. time elapsed: ', batch_end-batch_start)
            
            # the variable data contains an entire batch of inputs and their associated labels

            #samples, labels = data
            #print('sending data to device')
            device_start = timer()
            samples, labels = samples.to(device), labels.to(device)  # Sends the data to the GPU
            device_end = timer()
            #print('data sent. elapsed time', device_end-device_start)

            #print("zeroing grad")
            optimizer.zero_grad()  # Zeroes the gradient, otherwise it will accumulate at every iteration
            # the result would be that the network would start taking huge parameter jumps as training went on
            #print('grad zeroed')

            #print('inferring...')
            infer_start = timer()
            output = model(samples)  # Forward passes the input data
            infer_end = timer()
            #print('inferred')
            #print('time elapsed during inference:', infer_end - infer_start)

            #print('computing loss')
            loss_start = timer()
            loss = criterion(output, labels)  # Computes the error
            loss.backward()  # Computes the gradient, yielding how much each parameter must be updated
            loss_end = timer()

            #print('updating weights')
            weights_start = timer()
            optimizer.step()  # Updates each parameter according to the gradient
            weights_end = timer()
            #print('weights updated. time elapsed: ', weights_end-weights_start)

            running_loss = loss.item()
            print('running loss', running_loss)
            '''if index % 10 == 9:
                print('[%d %5d] loss %.3f' % (epoch + 1, index + 1, running_loss / 2000))
                running_loss = 0.0'''
            #print('loading new batch')
            batch_start = timer()

            image_qtt += samples.size()[0]
            images_since_last_save +=  samples.size()[0]
            if(images_since_last_save > 500):
                print('saving checkpoint', image_qtt)
                save_model(model, epoch, image_qtt, optimizer, 'custom_fcn_'+epoch+'_'+str(image_qtt))

    print('finished training')

#Saves the model as well as information related to training, so it can be resumed later
def save_model(model, epoch, image_index, optimizer, filename):
    checkpoint = {'state_dict': model.state_dict(), 'epoch': epoch, 'image_index': image_index, 'optimizer': optimizer}
    with open(filename, 'wb') as file:
        pickle.dump(checkpoint)

#Loads a model with the specified filename
def load_model(filename)
    path = 
    if(os.path.exists('checkpoints\\'+filename)):
        checkpoint = {}
        with open(filename, 'rb') as file:
            checkpoint = pickle.load(file)
        return checkpoint['state_dict'], checkpoint['epoch'], checkpoint['image_index'], checkpoint['optimizer']
    raise Exception('File not found')

#Legacy code used to test model validation, using an alexnet fine-tuned to the CIFAR-10 dataset
def validate(model, test_dataset, device):
    correct = 0
    total = 0
    model.eval()  # Sets the flag for evaluation
    # ensures batch normalization and dropout layers will stay inactive
    with torch.no_grad():
        for data in test_dataset:
            # loads a batch of data, sends it to GPU
            samples, labels = data
            samples, labels = samples.to(device), labels.to(device)

            # Forward passes the data
            outputs = model.forward(samples)

            # Takes the highest value from the output vector, computes the accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network: %d %%' % (100 * correct / total))
    return correct, total

#Auxiliary function to plot a Pytorch tensor
def plot_tensor(tensor):
    plt.imshow(transforms.ToPILImage()(tensor), interpolation="bicubic")
    plt.show()

def mask_to_color(net_output):
	return None

# Main code for training (still using legacy code for alexnet finetuning and classification)
# Presently under modification
def main():
    # sets up cuda
    print("Cuda availability status:", torch.cuda.is_available())
    print('setting device...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device set')
    #device = torch.device("cpu")

    print('loading alexnet')
    alexnet = models.alexnet(pretrained=True)
    print('alexnet loaded')
    
    glob


    print('loading cnn')
    cnn = CustomFCNAlexnet(num_classes=5, alexnet=alexnet)
    print('cnn loaded')
    print('sending cnn to gpu')
    cnn = cnn.to(device)
    print('gpu transfer successful')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    coco_api = COCO("coco\\ground_truth\\instances_train2014.json")
    print('creating dataset')
    coco_dataset = CocoDataset("coco\\images", coco_api, transform)
    print('dataset created')

    print('creating loader')
    train_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=8, shuffle=False)
    print('loader created')

    print('training')
    fit(cnn, train_loader, device)

    torch.save(cnn.state_dict(), 'cnn.pt')

    return 0


if __name__ == '__main__':
    main()
