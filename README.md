# Asset-Maintenance
Asset maintenance is the process of collecting Data, analysing data to make better decision for Assets in a particular organization. It plays a major role in manufacturing operations. With time physical assets of an organization such as pipes, machinery, vents etc, deplete over time decreasing their estimated life span. Thus, for them to work till the estimated time process of maintenance is essential, and most important sub part of its detection of that irregularity produced due to natural processes.

# Convolutional Neural Networks (CNN)
As the field of Artificial intelligence has flourished the need for machines to perceive world as human do has emerged overtime (Computer Vision). There are many algorithms that helps Machine to perceive the world as humans do but CNN- convolutional neural networks are one of the best ways since other classification methods are hand tailored on the contrary CNN can learn these methods or filters on its own to classify images. It is inspired by human visual cortex or how human brain processes images.
It assigns weights to various aspects of images thus, creating a framework to differentiate them. CNN is based on Grid that means it process data that is based on Grid Pattern, CNN later assign weights, based on priority (high to low) of an aspect or feature in that image.

![image](https://github.com/user-attachments/assets/87bbfc15-b80c-4d83-ba36-c697e597c236)
### Fig: Implementation of CNN on Mnist Dataset

Parameter- variable that is automatically learned during training process. (weights)
Hyperparameter- that must be set before training process takes place.
Kernel-set of learnable parameters in a model.
In Digital images pixels are stored as a 2D-linear structure such as Array thus, forming a Grid of n x m size where n is number of rows and m are number of columns. Where each block is an aspect or feature of Image
Layers in Convolutional neural networks:
•	Convolutional
•	Pooling
•	Fully connected layers.
Where Convolutional and Pooling Layers does data pre-processing that is in this case Feature extraction for dimensionality reduction. Third layer i.e., is fully connected layer maps features in the final output.

# ResNet:
With the increase in number of layers in CNN, training error was increased thus leading to vanishing or exploding gradient problem.

![image](https://github.com/user-attachments/assets/9f4b5a6f-79de-4663-bad0-dfa5bdcd1997)

## Vanishing/exploding gradient problem-
This causes gradient to become 0 or too large (repeated multiplication may make gradient infinitely small)
Thus, increasing training and test error rate
In a 56-layer CNN, error is more than 26-layer CNN.
## Batch Normalization-
Larger inputs cause instability leading to exploding gradient problem
Sometimes one weights gets larger than others, thus causing instability 
Thus, we need batch normalization.

As increasing the layers in NNs led to increase in accuracy and decrease in training error, question arises that if stacking more layers to the NNs will lead to better performance?

Results said otherwise since increasing the layers or depth of neural network led to the problem of vanishing gradient or exploding gradient.
With increase in layers accuracy gets saturated thus increasing training error. Therefore, residual networks were introduced that works on the principal of skip connections (or shortcut connections), accuracy of residual networks (or ResNet) was found to be greater than its counterpart that is plain networks. Thus, a very low training error of about 3.57%
- Uses skip connections – skips training from few layers and connects directly to output
- Advantages – if any layer hurt the performance of the architecture, then it will be skipped by regularization
- It used parameterized gates
- Uses batch normalization after each convolutional and before activation
- Uses He initializer
- Batch size 26
- Learning rate is 0.1 and is divided by 10 when error plateaus
- Trained for 60 x 10^4 iterations
- Weight decay of 0.0001, momentum of 0.9
- Does not use dropout

  ![image](https://github.com/user-attachments/assets/12f7da4c-9c67-461c-912d-89c4d62703d1)
  ### Fig: A Residual Network

# Methodology
## Description
In this project our motive is to classify images in Rust and No Rust categories that is binary classification using Transfer Learning.
The project is focused on detecting rust as inconsistency in any given piece of a pipeline or an object. Feeding images to trained ResNet classifies the object as Rust or No Rust. To maintain originality of data, we have collected our own dataset from industrial areas, public areas etc to dataset consists of two type of images that is namely Rust and No rust. Data Pre-processing has been performed over collected dataset, for feeding it to ResNet, for data reprocessing libraries like OpenCV, TensorFlow, keras has been used, thus the created training dataset is fed to ResNet to attain accuracy of the Convolutional Neural Network.
Project Consists of:
1.	Collection of Dataset
2.	Data Pre-processing
3.	Selecting right CNN architecture 
4.	Implementing CNN Architecture using Transfer Learning


## Collection of Dataset
Artificial Neural Networks depends heavily on Data Collection, without data we will not be able to train our Neural network to make expected predictions, thus a good dataset for training is very important with minimum errors as possible.
Collection of data is not enough but proper labelling is essential to for NN to understand the inconsistency, thus we have used os, matplotlib, np, cv2 library for proper labelling and collected images.

Some Sample images are given below -


![image](https://github.com/user-attachments/assets/307ef422-984f-4f5b-9a8f-7107da097460) ![image](https://github.com/user-attachments/assets/c2457484-648d-486b-94b2-90235c4e2141)

![image](https://github.com/user-attachments/assets/3acbee88-7369-4acb-8ebd-564bd9c2203f)

### Fig: Samples of Rusted images



![image](https://github.com/user-attachments/assets/9625ddaa-3a61-4492-92a2-ac26221d463d)![image](https://github.com/user-attachments/assets/66ce183d-e758-4542-bf41-c5f70e43cb0c)

![image](https://github.com/user-attachments/assets/9019c752-bc69-4b16-bbbb-6c3cce02b9cc)

### Fig: Samples of Non-Rusted Images

## Data Pre-processing
Data must be pre-processed to make it suitable for using as training data or test data, unprocessed data has chances that data may contain error or can be inconsistent thus to make it suitable we have to use certain methods or libraries to make it suitable. 
Here we have used libraries such as os, matplotlib, np, cv2 to make data fit for training Artificial Neural Network.
It involves steps such as
•	Converting collected images to a certain size that is required by ResNet
•	Attaching Labels to the images to i.e., Rust and No rust.

![image](https://github.com/user-attachments/assets/02c57e63-7b90-44ec-a593-50966c6a58a2)
### Fig: Pre-Processed image

## Function for Image Pre-processing and Attaching Labels

### ![image](https://github.com/user-attachments/assets/8a9ac6b3-42f9-4c19-bb2c-d6605c48e12d)

Given categories to add labels to the Images fed to the function create_training _data.


### ![image](https://github.com/user-attachments/assets/c76c745c-83f3-413b-86d5-5c4daf37e216)
Array Attained after pre-processing

### ![image](https://github.com/user-attachments/assets/c47c19f4-d9ed-4ceb-bcb7-18687504b854)
## Implementing CNN Architecture using Transfer Learning
We choose ResNet Architecture to implement our asset maintenance project as ResNet or Residual Networks can stack many layers without explosive gradient problem.
Here we used libraries such as matplotlib, NumPy, PIL, TensorFlow, keras to implement as Residual Network using Transfer learning.
In Transfer Learning we implement predesigned and pretrained Neural Network adding 2-3 layers at the last thus providing suitable classification at the end.


### ![image](https://github.com/user-attachments/assets/dc6d6691-412f-4b6c-9bdf-19307fafff32)
Fig: Implementation of ResNet using Transfer Learning

### ![image](https://github.com/user-attachments/assets/acc0aa8a-dade-4c2b-a865-6b5d38fe90bb)
### Fig: Summary of Implementation
Images are fed into network with a Size of 100x100, 3 here represents that image are coloured, classes=2 represents that the whole network is designed for binary classification of images i.e., Rust and No rust.
We used a Network with an epoch of 50, we used optimizer as Adam and a validation split of 0.1 or 10%


### ![image](https://github.com/user-attachments/assets/7281fb08-61de-4103-9bdb-0003d2a7645f)

### Fig: Training of ResNet

We acquired an accuracy of 84.7 %.

### ![image](https://github.com/user-attachments/assets/4728c463-4890-4fc2-9fcd-8b7f3987a135)
# Conclusion
In the following paper we have discussed about asset maintenance project for classification of images with inconsistencies i.e., in this case were rusted images. We have discussed about methodology for the project that provides us with working as well as the libraries used in this project. 
Asset maintenance is an emerging need in many industries thus automating the process provides us with faster detection of inconsistencies, hence delivering faster response time against the fault found. This project with an 84.7 % accuracy proves that Neural Networks can be used in this sector to deliver less fault detection time, they can be trained on huge datasets to provide us with better results. Improving their performance can be done in various ways, using many other strategies like increasing the size of training dataset, increasing size of provided coloured images etc.
In future many more changes can be done, and this framework can be integrated in form of application to be deployed.

# References
•	www.ibm.com
•	www.artesis.com
•	www.towardsdatascience.com
•	Yamashita, R., Nishio, M., Do, R.K.G. et al. Convolutional neural networks: an overview and application in radiology.
•	www.kaggle.com
•	A Transfer Residual Neural Network Based on ResNet‐34 for Detection of Wood Knot Defects Mingyu Gao, Jianfeng Chen, Hongbo Mu and Dawei Qi 
•	Teja Kattenborn, Jens Leitloff, Felix Schiefer, Stefan Hinz,
•	Review on Convolutional Neural Networks (CNN) in vegetation remote sensing
•	Deep Residual Learning for Image Recognition Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun Microsoft Research
•	Yamashita, RikiyaNishio, MizuhoDo, Richard Kinh GianTogashi, Kaori, Convolutional neural networks: an overview and application in radiology



