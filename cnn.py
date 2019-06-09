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
            print('sending data to device')
            device_start = timer()
            samples, labels = samples.to(device), labels.to(device)  # Sends the data to the GPU
            device_end = timer()
            print('data sent. elapsed time', device_end-device_start)

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
            loss_start = timer()
            loss = criterion(output, labels)  # Computes the error
            loss.backward()  # Computes the gradient, yielding how much each parameter must be updated
            loss_end = timer()
            print('loss computed. time elapsed: ', loss_end-loss_start)

            print('updating weights')
            weights_start = timer()
            optimizer.step()  # Updates each parameter according to the gradient
            weights_end = timer()
            print('weights updated. time elapsed: ', weights_end-weights_start)

            running_loss = loss.item()
            print('running loss', running_loss)
            '''if index % 10 == 9:
                print('[%d %5d] loss %.3f' % (epoch + 1, index + 1, running_loss / 2000))
                running_loss = 0.0'''
            print('loading new batch')
            batch_start = timer()
            

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
        print('loading image')
        imload_start = timer()
        image = Image.open(self.image_dir + "\\" + self.imgs[i]['file_name'])
        if image.mode == 'L':
            #print('converting gray scale to RGB')
            image = image.convert('RGB')
        image = self.transform(image)
        imload_end = timer()
        image = self.pad_image(image, 800, 800)
        print('input image loaded. elapsed time:', imload_end-imload_start)

        print('loading gt')
        gtload_start = timer()
        gt = self.load_ground_truth(i, 800, 800)
        gtload_end = timer()
        print('gt loaded. elapsed time:', gtload_end-gtload_start)
        
        print('gt ops')
        gtops_start = timer()
        
        '''transpose_start = timer()
        gt = np.transpose(gt, (0, 1, 2))
        transpose_end = timer()
        print('transposed elapsed time', transpose_end-transpose_start)'''
        
        cast_start = timer()
        gt = torch.tensor(gt, dtype=torch.long)
        cast_end = timer()
        print('cast time', cast_end-cast_start)

        #pad_start = timer()
        #gt = self.pad_image(gt, 800, 800)
        #pad_end = timer()
        #print('pad time', pad_end-pad_start)
        
        gtops_end = timer()
        print('gtops. elapsed time:', gtops_end-gtops_start)

        return image, gt[0,:,:]

    def __len__(self):
        return len(self.imgs)

    def load_ground_truth(self, i, desired_height, desired_width):
        annIds = self.coco.getAnnIds(imgIds=self.imgs[i]['id'], iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        seg_imageNch = np.zeros((len(self.classes), desired_height, desired_width)).astype(np.uint8)
        # seg_imageGray = np.zeros((self.imgs[i]['height'], self.imgs[i]['width'])).astype(np.uint8)
        for i in range(len(anns)):
            if anns[i]['category_id'] in self.classes.keys():
                seg_image = self.coco.annToMask(anns[i]).T
                seg_imageNch[self.classes[anns[i]['category_id']],:seg_image.shape[0],:seg_image.shape[1]] = \
                seg_imageNch[self.classes[anns[i]['category_id']],:seg_image.shape[0],:seg_image.shape[1]] | seg_image[:,:]
                # seg_image = (seg_image - (seg_image & seg_imageGray))
                # seg_imageGray = (seg_imageGray + ((seg_imageGray | seg_image) == 1) * self.classes[anns[i]['category_id']])
        # seg_imageNch[:, :, 0] = seg_imageGray.astype(np.uint8)
        return seg_imageNch

    def pad_image(self, source, desired_height, desired_width):
        padded_image = torch.zeros(source.shape[0], desired_height, desired_width).type(source.type())
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
    train_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=16, shuffle=False)
    print('loader created')

    print('training')
    fit(cnn, train_loader, device)

    torch.save(cnn.state_dict(), 'cnn.pt')

    return 0


if __name__ == '__main__':
    main()
