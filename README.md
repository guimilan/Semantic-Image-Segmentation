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
	5. File structure
	6. Results
	7. Conclusion

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

Next, it is necessary to take into account that the backpropagation method used for training is based on the gradient descent optimization technique, which requires a loss function to be defined. Given that the softmax layer outputs a set of probability distributions and the ground truth provides the real probability distribution expected for each correctly classified pixel, the Cross-entropy function was chosen as the loss. Cross-entropy compares the amount of information necessary to encode events of a hypothetical probability distribution with the expected amount of information necessary to encode information from a reference probability distribution. The lower the cross-entropy, the more similar the two distributions are, thus making it an appropriate optimization target. 

Finally, it is necessary to find ways to measure model accuracy over the task of image segmentation. Fidelity of the segmentation mask to the ground truth used for training can be assessed by computing the intersection-over-union (IoU) of each segmentation mask produced at the output of the network to the one contained in the ground truth. 

## 3. Image sets

For the purpose of training the network, the MS Coco dataset will be used [1]. It contains a wide range of
pictures displaying common objects, such as the sample below, as well as the ground truth containing each image's  segmentation masks. 

<img src="http://images.cocodataset.org/train2014/COCO_train2014_000000002758.jpg" alt="MS COCO sample" width="200" height="200"><img src="https://content.screencast.com/users/romeubertho/folders/Snagit/media/1cefcc80-503a-4207-8df4-1927a2f801ab/05.29.2019-23.12.png" alt="MS COCO segmentation sample" width="200" height="200">

Sample image from MS Coco

It is expected that the image quantity and diversity will enable the model to capture general features. This should, in turn, allow fine tuning on smaller datasets to achieve reasonable performance. Two such datasets include Polyp-7, comprised of 
segmented medical images by the Computer Vision Center (CVC) of Barcelona[2], and CamVid[3], which provides segmented videos and still images of public streets during daylight.


<img src="https://i.ibb.co/NNk3Vyf/polyp.png" alt="Polyp sample" width="200" height="200">

Medical image sample from the Polyp-7 dataset

<img src="https://i.ibb.co/G3xZV6S/street-segment.png" alt="CamVid sample" width="200" height="200">

Public street image sample from the CamVid dataset

## 4. Development

The Python programming language along with the PyTorch framework were chosen for the task of implementing an Alexnet-based FCN. 

## 3. Progress report

Work started with a survey of existing semantic segmentation models. The FCN-based architectures were deemed the most reasonable for the proposed deadline, both due to the relative ease of comprehension and implementation, as well to the computational power required in comparison with more recent models. 

FCN-based models have been developed on top of several different architectures, notably AlexNet and VGG. For this work, the chosen approach was to take a pretrained Alexnet, and gradually add the layers that turn it into a FCN.

Up to this point, a pretrained Alexnet has been successfully loaded, its layers frozen and fine tuned for classification on a smaller dataset with an accuracy rating of 85%, which is about the reported accuracy value for the Alexnet. Most recently, the classification layers were removed to make way to the first deconvolution layers. Some of the legacy code in which the classification dataset was loaded is still present and undergoing modification. The most recent iteration can be found on the file "cnn.py".

COCO provides its segmentation masks in a compressed, encoded format. Performing decoding and coupling each mask with its corresponding image during training would result in additional computational effort to an already intensive task. Therefore, several preprocessing strategies were attempted. Initially, focus was set on trying to maintain the maximum amount of information from the original dataset as possible while trying to minimize the computational effort required to load the minibatches during training. 

The first attempt involved generating one-channel images, in which each pixel had the value corresponding to its class id as annotated in the ground truth. During training, each minibatch would expand the images into 90-channel images, 90 being the total number of classes provided by Coco. After expansion, each pixel on each channel would contain a binary value, with 1 meaning the pixel on location (x,y) and channel k belongs to class k, and a 0 meaning it doesn't. Only a single channel would contain a value of 1 for a pixel, while all the others would show 0. 

The second approach consisted of generating plain text files for each image, which would contain a matrix for every object category featured in the respective image. Those matrixes would represent the channels of the 90-channel image used during training, in which a value of 1 at pixel (x,y) and matrix k would mean that that pixel belongs to class k. Matrixes which had only null values were not stored and their absence would imply a null matrix. Metadata contained in the files would inform which class each matrix belongs to.

Through experimentation, those approaches were deemed too time-consuming to perform during training, so it was decided that our scope had to be narrowed so as to make training viable on the available hardware. Therefore, experiments will now be performed with a small subset of the classes featured on MS Coco. 

Planned next steps include writing the code for training on MS Coco, which should complete the basic data pipeline that would allow proper experimentation to begin. This will be followed by tweaks to the architecture and fine tuning to specific tasks, such as medical image and street image segmentation. Finally, training on VGG-based FCNs (FCN-32, FCN-16 and FCN-8) will be tested, provided there's enough time left.

## References
[1]COCO - Common Objects in Context. Available on http://cocodataset.org/

[2]A benchmark for Endoluminal Scene Segmentation of Colonoscopy Images. Available on https://arxiv.org/pdf/1612.00799.pdf

[3]Motion-based Segmentation and Recognition Dataset. Available on mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

[4]Fully Convolutional Networks for Semantic Segmentation. Available on https://arxiv.org/abs/1411.4038

[5]Torchvision.models documentation. Avaible on https://pytorch.org/docs/stable/torchvision/models.html

[6]Segmentation and Recognition Using Structure from Motion Point Clouds, ECCV 2008
Brostow, Shotton, Fauqueur, Cipolla

[7]Semantic Object Classes in Video: A High-Definition Ground Truth Database
Brostow, Fauqueur, Cipolla 
