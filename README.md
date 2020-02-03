# Annoushka
A TensorFlow-inspired ANN library in pure ES6 without _any_ external dependencies.

## Code overview
### `ann`
The main library. Implemented Linear, Sigmoid and MSE nodes.

ReLU and Softmax activation, as well as Cross Entropy cost function, are planned.

The end-goal here is to have a network that decently performs on the MNIST dataset.

### `math`
Some necessary minimalistic maths stuff. Mostly linear algebra.

### `boston_demo`
An end-to-end example of deploying the Ann to work out the Boston Housing Prices.

## Running it
```
$ yarn
$ ./node_modules/.bin/jest .
$ node --experimental-modules boston_demo.mjs
```