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


class CNN(nn.Module):
    # Alexnet-based FCN implementation
    # for the sake of simplicity, at first the input will have an assumed dimension
    # of 227x227x3
    def __init__(self, num_classes=90, alexnet=None):
        super(CNN, self).__init__()

        # Convolution layers for feature extraction
        # Sizes: 227x227, 55x55, 27x27, 13x13 e 6x6
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
        for param in self.parameters():
            param.requires_grad = False

        # size-1 convolution for pixel-by-pixel prediction
        self.score_conv = nn.Conv2d(256, num_classes, 1)

        # Deconvolution layers for restoring the original image
        # input: 6x6x90, output: 13x13x256
        self.deconv1 = nn.ConvTranspose2d(num_classes, 256, kernel_size=3, stride=2)

        # input: 13x13x256 (skip-connect to conv5's output), output: 27x27x64
        self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2)

        # input: 27x27x64 (skip-connect to conv1's output), output: 55x55x64
        self.deconv3 = nn.ConvTranspose2d(64, 192, kernel_size=3, stride=2)

        # input: 55x55x64, output: 227x227xnum_classes
        self.deconv4 = nn.ConvTranspose2d(192, num_classes, kernel_size=8, stride=4)

    def forward(self, x):
        print('input shape', x.size())
        # Forward passes the data
        # Skip connections are formed by summing together the two connected layers' output
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
        # print('deconv 4 output shape', out_deconv4.size())
        return out_deconv4


def fit(model, train_dataset, device):
    criterion = nn.CrossEntropyLoss()  # Error function: cross entropy
    print('instantiating optimizer')
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print('optimizer instantiated')
    # Stochastic gradient descent
    # Initial learning rate: 0.001
    # momentum: 0.9

    for epoch in range(2):
        running_loss = 0.0
        model.train()  # Sets a flag indicating the code that follows performs training
        # this makes sure the dropout and batch normalization layers perform as expected

        for index, data in enumerate(train_dataset, 0):
            print('new batch')
            batch_start = timer()
            # the variable data contains an entire batch of inputs and their associated labels

            samples, labels = data
            print('just loaded data. samples data type', samples.type())
            print('labels type', labels.type())
            samples, labels = samples.to(device), labels.to(device)  # Sends the data to the GPU
            print('after sending to GPU. samples data type', samples.type())
            print('labels type', labels.type())

            print("zeroing grad")
            optimizer.zero_grad()  # Zeroes the gradient, otherwise it will accumulate at every iteration
            # the result would be that the network would start taking huge parameter jumps as training went on
            print('grad zeroed')

            print('inferring...')
            infer_start = timer()
            output = model(samples)  # Forward passes the input data
            infer_end = timer()
            print('inferred')
            print('time elapsed during inference:', infer_end - infer_start)

            print('computing loss')
            print('output type', output.type())
            print('labels type', labels.type())

            print('output shape', output.size())
            print('labels shape', labels.size())

            loss = criterion(output, labels)  # Computes the error
            loss.backward()  # Computes the gradient, yielding how much each parameter must be updated
            print('loss computed')

            print('updating weights')
            optimizer.step()  # Updates each parameter according to the gradient
            print('weights updated')

            running_loss += loss.item()
            print('running loss %.3f', running_loss)
            if index % 10 == 9:
                print('[%d %5d] loss %.3f' % (epoch + 1, index + 1, running_loss / 2000))
                running_loss = 0.0
            batch_end = timer()
            print('time elapsed for batch processing', batch_end - batch_start)

    print('finished training')


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


# Class representing the dataset
class CocoDataset(Dataset):
    def __init__(self, image_dir, coco, transform):
        self.image_dir = image_dir
        self.coco = coco
        self.allImgIds = None
        self.classes = None
        self.selectClass()
        self.imgs = coco.loadImgs(self.allImgIds)
        self.transform = transform

    def __getitem__(self, i):
        # print('loading image')
        image = Image.open(self.image_dir + "\\" + self.imgs[i]['file_name'])
        if image.mode == 'L':
            print('converting gray scale to RGB')
            image = image.convert('RGB')
        image = self.transform(image)

        image = self.pad_image(image, 800, 800)

        gt = self.load_ground_truth(i)
        gt = self.transform(np.transpose(gt, (0, 1, 2))).type(torch.LongTensor)
        gt = self.pad_image(gt, 800, 800)

        return image, gt

    def __len__(self):
        return len(self.imgs)

    def load_ground_truth(self, i):
        annIds = self.coco.getAnnIds(imgIds=self.imgs[i]['id'], iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        seg_imageNch = np.zeros((self.imgs[i]['height'], self.imgs[i]['width'], len(self.classes))).astype(np.uint8)
        # seg_imageGray = np.zeros((self.imgs[i]['height'], self.imgs[i]['width'])).astype(np.uint8)
        for i in range(len(anns)):
            if anns[i]['category_id'] in self.classes.keys():
                seg_image = self.coco.annToMask(anns[i])
                seg_imageNch[:, :, self.classes[anns[i]['category_id']]] = seg_imageNch[:, :, self.classes[anns[i]['category_id']]] | seg_image
                # seg_image = (seg_image - (seg_image & seg_imageGray))
                # seg_imageGray = (seg_imageGray + ((seg_imageGray | seg_image) == 1) * self.classes[anns[i]['category_id']])
        # seg_imageNch[:, :, 0] = seg_imageGray.astype(np.uint8)
        return seg_imageNch

    def pad_image(self, source, desired_height, desired_width):
        padded_image = (-1) * torch.ones(source.shape[0], desired_height, desired_width).type(source.type())
        padded_image[:, :source.size()[1], :source.size()[2]] = source

        return padded_image

        # Custom collating function. Used to combine individual tensors into a single batch
        # This was made to replace pytorch's default collate during debugging

    def collate(self, batch):
        print('collating')
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        print('collating successful')
        return data, target

    def selectClass(self, start=0, end=5):
        Cats = {}
        for i in range(len(self.coco.catToImgs)):
            Cats[i] = len(self.coco.catToImgs[i])
        Cats = list(sorted(Cats.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
        allImgIds = []
        catIndex = 0
        catToClass = {}
        for f in Cats[start:end]:
            catToClass[f[0]] = catIndex
            allImgIds = list(set(allImgIds) | set(self.coco.catToImgs[f[0]]))
            catIndex = catIndex + 1
        self.allImgIds = allImgIds
        self.classes = catToClass
        return self.allImgIds, self.classes


def plot_tensor(tensor):
    plt.imshow(transforms.ToPILImage()(tensor), interpolation="bicubic")
    plt.show()


# Main code for training (still using legacy code for alexnet finetuning and classification)
# Presently under modification
def main():
    # sets up cuda
    print("Cuda availability status:", torch.cuda.is_available())
    print('setting device...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device set')
    device = torch.device("cpu")

    print('loading alexnet')
    alexnet = models.alexnet(pretrained=True)
    print('alexnet loaded')

    print('loading cnn')
    cnn = CNN(num_classes=5, alexnet=alexnet)
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
    train_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=8, shuffle=False, num_workers=1)
    print('loader created')

    print('training')
    fit(cnn, train_loader, device)

    '''print('loading batch')
    for index, data in enumerate(train_loader):
        print('batch num', index)
        sample, label = data
        print('sending batch to gpu')
        sample = sample.to(device)
        label = label.to(device)
        print('batch sent to gpu')
        print('forward passing')
        out = cnn(sample)
        print('forward passed. result shape', out.size())
        if(index == 10):
            break'''

    '''
    #loads cifar10 data
    print("loading dataset")
    train, test, classes = load_cifar10(16)

    #instantiates pretrained alexnet
    print("dataset loaded. instantiating alexnet...")
    alexnet = models.alexnet(pretrained=True)
    print('alexnet instantiated. instantiating model...')
    cnn = CNN(10, alexnet)

    #Sends the network to the GPU (if one if available. otherwise, retains it on the CPU)
    print('done. sending model to gpu')
    cnn.to(device)
    print('successfully sent. begin training...')	
    
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
