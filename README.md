Independent-study
===================


### Week 1 - Probing Discrete Cosine Transfomation in Images to learn about the coherence/meaning of weights learned by Neural Nets for such transformations
1. Basic Done
    + Take a Dataset (eg. Mnist) , convert the images into dct-images, to get the input and output. (Store if time extensive computation)
    + Desgin a Basic Neural Network (1-hidden layer perceptron) with simple loss function (eg. mse) and inputs/outputs-flattend. 

2. High Level Experiments
    + Different Neural Networks
    + Multiple visualization Techniques

3. Issues
    + Loss function (As DCT preserves information in lower indices, shouldnt the loss functions be designed accordingly)
    + Data-Augmentations is possible (while training)?

4. Results and Further Discussion
	+ Mnist Dataset, MSE loss and a few hidden layer were tried upon in a fully-connected architecture.
    + Only first quadrant of the DCT is effeciently captured while minimizing loss.
    + Data-Augmentations is not possible since DCT of the image will change
    + Visualizing the weights of the learned Neural Networks do not provide any specific information
    + Example images depicting the same phenomenon can be seen [image1](week1/1.png) [image2](week1/2.png)
    + Lets forget about DCT and try to run an autoencoder.

### Week 2 - Probing encoder, decoder weights in autoencoder. Can these be transformed from one to another without learning one of them.
1. Basic Idea
	+ Train an Autoencoder for images (same image as input/output). Can these learned represenations be learned in half the weights?
	+ i.e. Train only the encoding part, can the decoding part be made just using the weights from the encoding part, with some simple transformation, manipulations.
	+ Advantages of the method being , effective training times

2. Experiments/Reading Material
3. Issues
	+
