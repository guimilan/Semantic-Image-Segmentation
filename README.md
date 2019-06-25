# Semantic Image Segmentation

**Guilherme Milan 9012966 (@guilhermesantos)**

**Romeu Bertho Jr 7151905 (@romeubertho)**

## Abstract
The aim of this project is to investigate the implementation and the applications of semantic segmentation via deep learning models. We will attempt to build and train a fully convolutional neural network (FCN) from scratch, as well as compare it to pretrained models and other CNN architectures. Finally, we will explore different applications, such as medical image segmentation and autonomous driving.

## Summary
	1. Development plan
	2. Theoretical foundation
	3. Image sets
	4. Development
		4.1 Building a data pipeline
		4.2 Defining the model's structure
		4.3 Data loading and preprocessing
	5. File structure
	6. Code excerpts
	7. Results
	8. Conclusion
	9. References

## 1. Development plan

Work will begin with a survey of existing semantic image segmentation algorithms. Once a technique is chosen, a study of its theoretical foundations will be conducted in order to provide a better grasp on the tasks to be accomplished, as well as to better sort through the tools and means available for the task. This will be followed with the selection of the appropriate programming language and libraries for implementation, testing and visualization of the resulting segmented images. In the condition that there is time remaining by the end of that process, competing models will be evaluated, allowing for the making of a comparative study based on academically accepted, objective metrics.

## 2. Theoretical foundation

Our study found that the image segmentation algorithms that produced the most promising results are those based on machine learning, namely neural network-based deep learning architectures. Notable examples include Fully Convolutional Networks (FCN), Mask R-CNN and Google's DeepLab. 

To the best of our knowledge, the FCNs were the earliest deep learning-based method proposed. Among the cited techniques, they are also the most straightforward, easy to comprehend and develop on the stardard libraries pertaining to the most popular deep learning frameworks. Thus, taking into consideration the timing and computing power constraints on this project, these networks were considered the most viable for implementation, and thereby were chosen as the starting point of our work.

There exist different varieties of fully convolutional networks that follow a common two-portion, layered structure. The first portion of which is formed by groups of convolutional layers, meant to extract features and reduce the dimensionality of the input image. The second portion, on the other hand, is based on layers that perform transposed convolutions. These layers restore the images' original dimension by applying a learneable filter to the first portion's output. In other words, they upsample the segmentation masks via parameters adjusted on the training data, which enables the networks to work on images of variable size. 

Some variations of the FCN also make use of an additional feature in the form of skip connections. Skip connections combine earlier, lower level feature maps produced on the initial layers to later, higher level features captured on the model's final layers. This is employed as a device meant to address the vanishing and exploding gradient problems presented by deeper networks. Respectively, this means the observed phenomenom of having gradients acquire exceedingly small or large values during backpropagation, impairing the initial layer's ability to learn from data. 

Skip connections are formed by either concatenation or summation of the earlier layer's output to the later layer's input. This is only possible if both feature maps share the same dimensionality, and therefore it is usually done between a convolution and a transposed convolution.

The network's final output consists of a k-channel image, where k is equal to the number of object classes the network is being trained to segment. The pixel value on each channel N consists of the probability of said pixel belonging to class N, thus, the values of a given pixel (x,y) on all the channels must sum to one. It is therefore advisable to make a channel-wise softmax layer the last layer in the network.

Having defined the architecture, it is then necessary to modify the model's parameters to fit the input data, which is usually called training. Training, particularly when it comes to deep learning, is a highly computationally intensive task which requires massive amounts of annotated data to accomplish. FCNs, however, exhibit the distinctive trait of having the aforementioned first portion be based on pre-existing networks, tipically the Alexnet and VGG. This has the advantage of enabling the model designer to incorporate layers from pre-trained instances of these networks into the FCN, in a practice known as fine tuning. As a result, the model as a whole requires less data and time to train. 

In this work, PyTorch's pretrained Alexnet was employed as the first portion of the network. It is important to note that the Pytorch implementation's architecture differs from the Alexnet used in the paper where the FCN was originally proposed, mainly in the number of convolutions per layer as well as the kernel size. This might prompt further study in which the original paper's approach could be tested against our model as a way to assess the impact of each adjustment in segmentation accuracy.

It is also of note that, due to the inclusion of skip connections between early and later layers, none of the pretrained layers were frozen, so that their weights are loaded as a form of initialization. The tranposed convolution layers are initialized randomly.

Next, it is necessary to take into account that the backpropagation method used for training is based on the gradient descent optimization technique, which requires a loss function to be defined. Given that the softmax layer outputs a set of probability distributions and the ground truth provides the real probability distribution expected for each correctly classified pixel, the Cross-entropy function was chosen as the loss. Cross-entropy compares the amount of information necessary to encode events of a hypothetical probability distribution with the expected amount of information necessary to encode information from a reference probability distribution. The lower the cross-entropy, the more similar the two distributions are, thus making it an appropriate optimization target. 

Finally, it is necessary to find ways to measure model accuracy over the task of image segmentation. Fidelity of the segmentation mask to the ground truth used for training can be assessed by computing the intersection-over-union (IoU) of each segmentation mask produced at the output of the network to the one contained in the ground truth. 

## 3. Image sets

For the purpose of training the network, the MS Coco dataset will be used [1]. It contains a wide range of
pictures displaying common objects, such as the sample below, as well as the ground truth containing each image's segmentation masks. MS Coco train dataset from 2014 has 82783 images, 13gb size and up to 80 classes labeled from 1 to 90. 

<img src="http://images.cocodataset.org/train2014/COCO_train2014_000000002758.jpg" alt="MS COCO sample" width="200" height="200"><img src="https://content.screencast.com/users/romeubertho/folders/Snagit/media/1cefcc80-503a-4207-8df4-1927a2f801ab/05.29.2019-23.12.png" alt="MS COCO segmentation sample" width="200" height="200">

Sample image from MS Coco

It is expected that the image quantity and diversity will enable the model to capture general features. This should, in turn, allow fine tuning on smaller datasets to achieve reasonable performance. Two such datasets include Polyp-7, comprised of 
segmented medical images provided by the Computer Vision Center (CVC) of Barcelona[2], and CamVid[3], which offers segmented videos and still images of public streets during daylight.


<img src="https://i.ibb.co/NNk3Vyf/polyp.png" alt="Polyp sample" width="200" height="200">

Medical image sample from the Polyp-7 dataset

<img src="https://i.ibb.co/G3xZV6S/street-segment.png" alt="CamVid sample" width="200" height="200">

Public street image sample from the CamVid dataset

## 4. Development

The Python programming language along with the PyTorch framework were chosen for the task of implementing an Alexnet-based FCN. The reasoning behind this decision comes from the fact that PyTorch follows an imperative, dynamic paradigm in which the computation graph is built during runtime. Thus, it provides simpler ways to interface regular Python code with the actual model building algorithms. This draws sharp contrast with competing libraries such as Tensorflow, that in its most common form requires static graph building, in which communication with the overarching program has to be made through the use of sessions and placeholders (eager execution notwithstanding), resulting in a steeper learning curve. Given that time is a strict constraint to be met, simplicity is a major factor to consider. Besides, PyTorch makes GPU computing simple, which is an absolute requirement for training large models, and it also provides data loading facilities that would lower the development overhead for building a complete training pipeline.

Following the theoretical study, work began with the initial definition of the model, as well as the experiment of loading a pretrained alexnet, replacing its fully connected layers and fine tuning it to a different dataset to the one it was trained on for the task of classification. The data came from the CIFAR-10 dataset, and a training pipeline was built according to PyTorch's guidelines.

### 4.1 Building a data pipeline

In accordance with the documentation, PyTorch has built-in tensor types that can be directly converted from numpy arrays. It builds tensors for the minibatches to be used in training in an automated way. For this purpose, it provides the Dataset class, which the developer must extend, or inherit from, in one's own class. Three methods must be implemented. The first is __init__, representing the class' constructor. The second method is __getitem__. This method loads a single (input, label) pair from the disk and performs whichever preprocessing the data requires. Finally, the method __length__ must be provided, which is supposed to return an integer describing the full length of the dataset.

With a Dataset subclass ready, it is possible to instantiate a DataLoader class, provided by PyTorch. This class takes a Dataset instance, a batch size and a boolean value representing whether it should shuffle the data. It is possible to iterate over the DataLoader, at every iteration of which it will return a pair of tensors (sample, labels) containing the full input and ground truth minibatches.

It is therefore possible to define the training loop as a for loop iterating over the DataLoader. Inside the training loop, it is possible to create an Optimizer object, which contains the optimization backend that provides functionality such as gradient computing and parameter updating. There are several Optimizer subclasses to choose from; for this work, stochastic gradient descent (SGD) was used. 

Having an optimizer, the input sample and the expected output, it is then possible to forward pass the data through the model, obtain the its prediction, and then compute the loss. To perform a forward pass, it is sufficient to provide an instance of the class in which the model is contained, ie, FCN. Calling the instance using the input data as an argument will perform the forward pass, ie, FCN(input_sample).

Pytorch also provides several loss functions, including cross entropy loss (in the form of the class CrossEntropyLoss). It is important to note that CrossEntropyLoss automatically applies softmax to the model's prediction, therefore making it unnecessary to explicitly add a softmax layer to the model.

Every error function provides the backward() method, which assesses the error and prompts the optimizer to compute the parameters' gradient. optimizer.step() has PyTorch update all the weights on the network based on those gradients, concluding the training loop.

### 4.2 Defining the model's structure

Pytorch offers a Module class, which abstracts a neural network's components in a way that allows them to be easily integrated into different architectures. In fact, entire neural networks can be embedded, as long as they subclass Module.

Thus, the proposed architecture is defined in the CustomFCNAlexnet class, which extends Module and progressively adds Alexnet's layers to its own. Layers have been grouped into Sequential modules, which consist of essentially atomic blocks that may wrap multiple operations, such as a convolution followed by rectified linear unit (ReLU) activation and max pooling, in a single function call. Two such blocks are stacked, followed by 3 sequentials comprised of convolutions followed by ReLU. A single scoring convolution is then added, after which 5 transposed convolution layers follow. Skip connections are performed during the forward pass by summing together the output of the connected layers.  

A second model was implemented in the class OriginalFCNAlexnet. It follows the original FCN architecture more closely, however its training loop was never fully implemented due to timing constraints. Since it is incompatible with PyTorch's pretrained model, all of its layers were randomly initialized. This was also a major factor in the decision to discontinue this line of work, considering the computational burden would lie beyond what this projects' resources could afford.

### 4.3 Data loading and preprocessing

COCO provides its segmentation masks in a compressed, encoded format. Performing decoding and coupling each mask with its corresponding image during training would result in additional computational effort to an already intensive task. Therefore, several preprocessing strategies were attempted. Initially, focus was set on trying to maintain the maximum amount of information from the original dataset as possible while trying to minimize the computational effort required to load the minibatches during training.

The first attempt involved generating one-channel images, in which each pixel had the value corresponding to its class id as annotated in the ground truth. During training, each minibatch would expand the images into 90-channel images, 90 being the total number of classes provided by Coco. After expansion, each pixel on each channel would contain a binary value, with 1 meaning the pixel on location (x,y) and channel k belongs to class k, and a 0 meaning it doesn't. Only a single channel would contain a value of 1 for a pixel, while all the others would show 0.

The second approach consisted of generating plain text files for each image, which would contain a matrix for every object category featured in the respective image. Those matrixes would represent the channels of the 90-channel image used during training, in which a value of 1 at pixel (x,y) and matrix k would mean that that pixel belongs to class k. Matrixes which had only null values were not stored and their absence would imply a null matrix. Metadata contained in the files would inform which class each matrix belongs to.
Through experimentation, those approaches were deemed too time-consuming to perform during training, so it was decided that our scope had to be narrowed to make training viable on the available hardware. Therefore, experiments will now be performed with a small subset of the classes featured on MS Coco.

Finally, the last approach consisted in the creation of a subset class from Pytorch’s Dataset class and instantiate a MS COCO API class in order to load images and generate ground truth for further usage at training stage. In order to archive that, it was provided the init, getitem and lenght methods to satisfy Pytorch standards. Loading and generating ground truth images in runtime significantly lowered the task time spent regarding the others approaches.

## 5. File structure
In this section, the content of each source file is indexed. Further documentation is available as commentary on each file.

### 5.1 ./main.py

__fit__: performs the main training loop

__save_model__: saves a model to disk in binary form into the ./checkpoints folder

__load_model__: looks up a model in the ./checkpoints folder and loads it from disk 

__load_latest_model__: parses the models in the ./checkpoints folder and loads the latest one, according to the 
naming convention adopted during saving 

__validate__: legacy code, originally used to validate a classification alexnet on the cifar-10 dataset

__norm__: normalizes an array

__plot_tensor__: plots a pytorch tensor as a matplotlib plot

__Norm__: normalizes an array

__load_cifar10__: legacy code, originally used to load the cifar-10 dataset used for the study of using pytorch for classification

__mask_to_color__: turns a segmentation mask into a colored, plottable image

__create_coco_data_loader__: loads a coco dataset instance and return a DataLoader that fetches input samples and ground truth from coco

__main__: runs the training loop

### 5.2./custom_fcn_alexnet.py

__class CustomFCNALexnet__: Class containing the definition of this work's customized FCN-Alexnet

__init__: Constructor that builds a custom FCN-Alexnet

__forward__: performs a forward pass through the network

### 5.3 ./coco_dataset.py

__class CocoDataset__: Class representing the Coco dataset. A subclass of PyTorch's dataset, it is used to feed the 
DataLoader that manages the training data

__init__: Constructor for CocoDataset instances

__getitem__: defines an instance's behavior when indexed by an integer, ie, instance[i], and must return an (input sample, ground truth) pair.  This method is used by the DataLoader to build each minibatch

__len__: returns an integer representing the dataset's total length. This is also required by the DataLoader

__load_ground_truth__: loads and returns the segmentation mask for an image

__pad_image__: pads an image to the specified size. PyTorch's minibatches requires every image tensor to have the same dimension, thus the need for this function.

__collate__: overrides the default PyTorch behavior for stacking several tensors into a single one. Used during debugging.

__selectClass__: returns the ids for all images that contain segmentation masks for the given class id range  

### 5.4 ./original_fcn_alexnet.py

__class OriginalFCNAlexnet__: Class containing a FCN-Alexnet based on the network proposed on original paper

__init__: Builds an instance of OriginalFCNAlexnet

__forward__: performs a forward pass through the network

### 5.5 ./train_original.py

__main__: script initially written to perform tests on the OriginalFCNAlexnet

### 5.6 ./Semantic Segmentation - Practical example.ipynb

This is the Jupyter notebook used to present the achieved results . It illustrates how to load the model, perform inference and visualize the results produced by the network

## 6. Code excerpts

Sample code of the Custom FCN-Alexnet  

<img src="https://content.screencast.com/users/romeubertho/folders/Snagit/media/6c29c990-5f84-44cc-b802-3708cd817cd9/06.24.2019-23.35.png" align="center">

Sample code of the CocoDataset responsible to handle dataset preprocessing. The methods below are required to satisfy Pytorch standards.

<img src="https://content.screencast.com/users/romeubertho/folders/Snagit/media/d13eaa10-58b1-484b-a3f8-6b466f1c831a/06.24.2019-23.24.png" align="center">

The following code loads the coco annotations as well as one of our trained FCN models, which is copied into the GPU. A PyTorch DataLoader to feed data to the model is also instantiated. Next, a small batch of data is loaded through the DataLoader and then proceeds to copy that data into the GPU where the network was stored. The model then performs inference on the data batch, which can be turned into a visible format.  

<img src="https://content.screencast.com/users/romeubertho/folders/Snagit/media/56ba360e-03ff-4d8a-9499-a9bad02110bd/06.24.2019-23.40.png" align="center">




## 7. Results

The results below consists in examples from the FCN-Alexnet that has been trained on a fraction of the whole COCO dataset (10k/87k available images). Although the training is incomplete, preliminiary predictions showed interesting results and surpassed the overall expectations.  

The original images and the generated masks are displayed side by side. The first column shows the original image, while the second column contains the raw output from the network. The third column displays the isolated mask, while the last column overlays the mask on the original image.

<img src="https://content.screencast.com/users/romeubertho/folders/Snagit/media/c9cae1b3-2a6f-425c-a57d-42e88e91f1c0/06.24.2019-20.59.png" alt="results" width="800" height="1400" align="center">

## 8. Conclusion
Due to the strict time constraints imposed on this project, not all experiments and measurements could be fully realized. However, the obtained results were considered satisfactory, with successful semantic segmentation being achieved with only a small subset of the entire database having been used for training. The architectured, thus, surpassed our initial expectations, and the originally proposed ideas remain as possible avenues for future study. 

## 9. References
[1]COCO - Common Objects in Context. Available on http://cocodataset.org/

[2]A benchmark for Endoluminal Scene Segmentation of Colonoscopy Images. Available on https://arxiv.org/pdf/1612.00799.pdf

[3]Motion-based Segmentation and Recognition Dataset. Available on mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

[4]Fully Convolutional Networks for Semantic Segmentation. Available on https://arxiv.org/abs/1411.4038

[5]Torchvision.models documentation. Avaible on https://pytorch.org/docs/stable/torchvision/models.html

[6]Segmentation and Recognition Using Structure from Motion Point Clouds, ECCV 2008
Brostow, Shotton, Fauqueur, Cipolla

[7]Semantic Object Classes in Video: A High-Definition Ground Truth Database
Brostow, Fauqueur, Cipolla 
