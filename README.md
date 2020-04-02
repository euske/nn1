# Convolutional Neural Network in C

 * Plain C99.
 * Simple.
 * Straightforward.
 * For educational purposes.

## `bnn.c`

 * Primitive SGD.
 * Only one type of layer (sigmoid).
 * No minibatch, no dropout, etc.

### Exercises

 * `$ cc -o bnn bnn.c -lm`
 * Increase the number of epochs and see how the error rate drops.
 * Try different learning rates.
 * Try different functions to learn.

## `cnn.c`

 * Simple SGD + Minibatch.
 * Three types of layers (input, convolutional, fully-connected).
 * ReLU for conv layers.
 * Tanh for non-last fc layers.
 * Softmax for the output (last) layer.

### Exercises

 * Obtain the MNIST database from http://yann.lecun.com/exdb/mnist/
 * Compile and run `mnist.c`.
 * Set the batch size to 1 (no minibatch) and see the results.
 * Try changing the last layer from softmax to tanh.
 * Change the network configurations and see how the accuracy changes.

## What I (re)discovered through this (re)implementation.

 * Use a proper learning rate.
 * Use minibatch training.
 * Use Softmax for the output layer.
 * Use Tanh/ReLU.
 * Choose the initial weight distribution wisely.
 * Feed the same data multiple times in a random order.
