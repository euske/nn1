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

## `rnn.c`

 * Stateful + Simple SGD + Minibatch.
 * Only one type of layer (Truncated BPTT).
 * No gate.

## What I (re)discovered through this (re)implementation.

 * Use a proper learning rate.
 * Use minibatch training.
 * Use Softmax for the output layer.
 * Use Tanh/ReLU.
 * Choose the initial weight distribution wisely.
 * Feed the same data multiple times in a random order.
 * Memorize past outputs/errors for RNN.

### What I further learned...

In addition to the usual ML tips (more data is better,
balanced data is better, data prep. matters, etc.),
I learned the following when you design your own NN model:

 * Encoding/decoding matters.
 * Loss function is super important. It can make or break the whole project.
 * Learning rate / scheduler is also important.
   (But today it's getting easier with Adam, etc.)
 * Do not mix different activation functions at the output layer.
