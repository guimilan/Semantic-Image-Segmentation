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

#Class representing the dataset
#This extends Pytorch's Dataset class so it can be used as a source for a 
#DataLoader object. It behaves like a Python collection by implementing
#methods such as length and __getitem__
class CocoDataset(Dataset):
    #The contructor must be passed the directory in which the images are contained, 
    #an instance of the COCOAPI object and the torchvision transforms to be applied
    #in the default case, this transform consists only of a PIL image to tensor
    #conversion (transforms.toTensor)
    def __init__(self, image_dir, coco, transform):
        self.image_dir = image_dir
        self.coco = coco
        self.allImgIds = None
        self.classes = None
        self.selectClass()
        self.imgs = coco.loadImgs(self.allImgIds)
        self.transform = transform

    #Defines the behavior of an instance of this class during indexing, ie, 
    #instance = CocoDataset()
    #first_dataset_image = instance[0]
    #PyTorch DataLoaders require this sort of functionality to be implemented
    def __getitem__(self, i):
        #print('loading image')
        imload_start = timer()
        image = Image.open(self.image_dir + "\\" + self.imgs[i]['file_name'])
        if image.mode == 'L':
            #print('converting gray scale to RGB')
            image = image.convert('RGB')
        image = self.transform(image)
        imload_end = timer()
        image = self.pad_image(image, 800, 800)
        #print('input image loaded. elapsed time:', imload_end-imload_start)

        #print('loading gt')
        gtload_start = timer()
        gt = self.load_ground_truth(i, 800, 800)
        gtload_end = timer()
        #print('gt loaded. elapsed time:', gtload_end-gtload_start)
        
        #print('gt ops')
        gtops_start = timer()
        
        '''transpose_start = timer()
        gt = np.transpose(gt, (0, 1, 2))
        transpose_end = timer()
        print('transposed elapsed time', transpose_end-transpose_start)'''
        
        cast_start = timer()
        gt = torch.tensor(gt, dtype=torch.long)
        cast_end = timer()
        #print('cast time', cast_end-cast_start)

        #pad_start = timer()
        #gt = self.pad_image(gt, 800, 800)
        #pad_end = timer()
        #print('pad time', pad_end-pad_start)
        
        gtops_end = timer()
        #print('gtops. elapsed time:', gtops_end-gtops_start)

        return image, gt[0,:,:]

    #Returns the size of this collection, which in this case means the number of images
    #in the dataset
    #This is also required by PyTorch's DataLoaders
    def __len__(self):
        return len(self.imgs)

    #Returns the segmentation mask for a specified image index
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

    #Pads an image to the specified size with zeroes. This is necessary, since all the images must have 
    #the same size in order to be loaded in a single minibatch tensor 
    #This also had to be done as a workaround to numpy's pad function, which for some reason
    #ran very slowly during training
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

    #Filters coco's images by class index
    #Selects only the classes in the index range [start, end]
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