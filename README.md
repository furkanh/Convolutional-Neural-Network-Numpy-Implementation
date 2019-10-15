Numpy implementation of Neural Network.
## Layers
You can create any neural network with this library. The code will create a computational graph as you add more layers. It will calculate the gradient using backpropagation.

You can train Convolutional Neural Network.

The design of the system is inspired from Keras and PyTorch libraries.

Note that this library does not use GPU so it will be slow.

This code may help new researchers to understand how high level deep learning libraries calculate gradients for complex operations e.g. convolution.

## Losses
Loss class extends Layer class in order to be able to calculate the gradient of the loss. It only contains CrossEntropy and L2Norm. L2Norm can be added to loss for regularization.

## Optimizers
Optimizers have step function to update weights of a given Network. SGD, Adadelta and Adam are implemented. You can extend Optimizer class to implement another optimizer.

## Initializers
Initializers have get_weights function. It returns random array with the given shape. GlorotUniform, GlorotNormal, HeNormal are implemented. You can implement another Initializer by extending Initializer class.
