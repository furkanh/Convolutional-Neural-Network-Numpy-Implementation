This is a Python library for creating and training neural network. It support convolutional neural networks as well. The design of the system is inspired by Keras, Tensorflow, and PyTorch libraries. This repository may help new beginners to understand how high level deep learning libraries calculate gradients for complex operations e.g. convolution.

## Layers
You can create neural networks with this library. The code will create a computational graph as you add more layers. It will calculate the gradient using backpropagation.

You can train Convolutional Neural Networks as well.

Note that this library does not use GPU so it will be slow.

## Losses
Loss class extends Layer class in order to be able to calculate the gradient of the loss. It only contains CrossEntropy and L2Norm. L2Norm can be added to loss for regularization.

## Optimizers
Optimizers have step function to update weights of a given Network. SGD, Adadelta and Adam are implemented. You can extend Optimizer class to implement another optimizer.

## Initializers
Initializers have get_weights function. It returns random array with the given shape. GlorotUniform, GlorotNormal, HeNormal are implemented. You can implement another Initializer by extending Initializer class.
