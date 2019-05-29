# Semantic Image Segmentation

**Guilherme Milan 9012966 (@guilhermesantos)**

**Romeu Bertho Jr 7151905 (@romeubertho)**

## Abstract

The aim of this project is to investigate the implementation and the applications of semantic segmentation via deep learning models. We will attempt to build and train a fully convolutional neural network (FCN) from scratch, as well as compare it to pretrained models and other CNN architectures. Finally, we will explore different applications, such as medical image segmentation and autonomous driving.

## 1. Image sets

For the purpose of training the network, the MS Coco dataset will be used [1]. It contains a wide range of
pictures displaying common objects, such as the sample below.

Sample 1
![GitHub Logo](https://ibb.co/3F8srCV)

It is expected that the image quantity and diversity will enable the model to capture general features. This should, in turn, allow fine tuning on smaller datasets to achieve reasonable performance. Two such datasets include Polyp-7, comprised of 
segmented medical images by the Computer Vision Center (CVC) of Barcelona[2], and CamVid[3], which provides segmented videos and still images of public streets during daylight.

Sample 2 (caption: Polyp-7 image sample), sample 3 (caption: CamVid)

## 2. Development plan

Studies will focus on the FCN architecture. This network constitutes of two main sections: convolution for feature extraction and deconvolution for generating the segmentation masks. In some versions of the network (FC-8 and FC-16), these two sections are also interconnected by skip layers. These are added with the goal of allowing the segmentation layers to produce finer masks by combining lower level, global information from earlier layers with fine-grained, local information from later layers.

For the first section of the network, the original paper[4] made use of the convolutional layers of a pretrained Alexnet. Thus, in order to save time, transfer learning will be performed, by taking and freezing the convolutional layers of such a network trained on the Imagenet dataset, like the one provided by torchvision (link). The rest of the network will then be trained on MS Coco. Subsequently, for the smaller datasets, only the final layer will be fine tuned instead of training the whole network again.

More modern implementations are built on top of the pretrained layers of a VGG network. This will be attempted as long as there is enough time left after the experiments with FCN-Alexnet have been performed.

Implementation will be done on PyTorch using CUDA.

## 3. Progress report

Work started with the survey of existing semantic segmentation models. The FCN-based architectures were deemed the most reasonable for the proposed deadline, due 

FCN-based models have been developed on top of several different architectures, notably AlexNet and VGG. The layers of a FCN-Alexnet have been implemented on PyTorch, as shown below.

Additionally, an Alexnet has been successfully loaded, its layers frozen and fine tuned for classification on a smaller dataset with good performance. The code to replace the FCN-Alexnet's default layers with the ones loaded from the pretrained Alexnet is already functional. 

The current step is to load the data from MS Coco's encoded format and present it to the network in a way that makes pixelwise prediction possible. 

Finally, the code for training on MS Coco will be written, and functionality for fine tuning for will be implemented as time allows.

References
[1]COCO - Common Objects in Context. Available on http://cocodataset.org/
[2]A benchmark for Endoluminal Scene Segmentation of Colonoscopy Images. Available on https://arxiv.org/pdf/1612.00799.pdf
[3]Motion-based Segmentation and Recognition Dataset. Available on mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
[4]Fully Convolutional Networks for Semantic Segmentation. Available on https://arxiv.org/abs/1411.4038