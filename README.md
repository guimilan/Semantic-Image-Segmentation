# Semantic Image Segmentation

**Guilherme Milan 9012966 (@guilhermesantos)**

**Romeu Bertho Jr 7151905 (@romeubertho)**

## Abstract

The aim of this project is to investigate the implementation and the applications of semantic segmentation via deep learning models. We will attempt to build and train a fully convolutional neural network (FCN) from scratch, as well as compare it to pretrained models and other CNN architectures. Finally, we will explore different applications, such as medical image segmentation and autonomous driving.

## 1. Image sets

For the purpose of training the network, the MS Coco dataset will be used (http://cocodataset.org/). It contains a wide range of
pictures displaying common objects, such as the sample below.

Sample 1

It is expected that the image quantity and diversity will enable the model to capture general features. This should, in turn, allow fine tuning on smaller datasets to achieve reasonable performance. Two such datasets include Polyp-7, comprised of 
segmented medical images by the Computer Vision Center (CVC) of Barcelona (link), and CamVid (mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/), which provides segmented videos and still images of streets during daylight.

Sample 2 (caption: Polyp-7 image sample), sample 3 (caption: CamVid)

## 2. Development plan

Studies will focus on the FCN architecture. This network constitutes of two main sections: convolution for feature extraction and deconvolution for generating the segmentation masks. In some versions of the network (FC-8 and FC-16), these two sections are also interconnected by skip layers. These are added with the goal of allowing the segmentation layers to produce finer masks by combining lower level, global information from earlier layers with fine-grained, local information from later layers.

For the first section, the original paper[citation] made use of the convolutional layers of a pretrained Alexnet. Thus, in order to save time, transfer learning will be performed, by taking and freezing the convolutional layers of such a network trained on the Imagenet dataset, like the one provided by torchvision (link). The rest of the network will then be trained on MS Coco. Subsequently, for the smaller datasets, only the final layer will be fine tuned instead of training the whole network again.

Implementation will be done on PyTorch using CUDA. 

## 3. Progress report

The layers of a FC-16 have been implemented on PyTorch. Additionally, an Alexnet has been successfully loaded, its layers frozen and fine tuned for a smaller dataset with good performance. The code to replace the FC-16's default layers with the ones loaded from the Alexnet is already functional. 

The next step will be to load the data from MS Coco's encoded format and present it to the network in a way that makes pixelwise prediction possible. Finally, the code for training on MS Coco will be written, and functionality for fine tuning will be implemented as time allows.
