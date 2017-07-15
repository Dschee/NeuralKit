# NeuralKit
<img src="https://raw.githubusercontent.com/palle-k/NeuralKit/master/Docs/badge.svg" alt="Documentation Status"/>

NeuralKit is a collection of 3 frameworks for training and running artificial neural networks.

## Contents

### MatrixVector
The MatrixVector framework provides a set of optimized SIMD and MIMD functions on vectors and matrices.

### NeuralKit
The NeuralKit framework provides an implementation of a feed forward neural network which runs on the CPU.

### NeuralKitGPU
The NeuralKitGPU framework provides a similar implementation of a feed forward neural network which runs on the GPU using Metal.

#### Serialization
The Serialization Framework includes helper functions used to store and load neural networks to and from JSON files.
(This framework will be replaced by the builtin Encoding and Decoding mechanisms in Swift 4)

## Functions

Both implementations provide many layers commonly used in neural networks:

- Fully connected layers
- Convolution layers
- Pooling layers
- Nonlinearity layers

Following activation functions are implemented:

- Tangens Hyperbolicus
- Sigmoid
- Rectified Linear Units
- Softmax (only for output layers)

Commonly used optimization methods are implemented:

- Vanilla Stochastic Gradient Descent
- Momentum SGD
- AdaGrad
- RMSprop
- AdaDelta


### Known Issues

- Networks using convolution layers cannot be optimized. Training currently diverges.
- Optimizers other than vanilla SGD and Momentum SGD currently lead to divergence of weights.
