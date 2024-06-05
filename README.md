# Neural Net Builder
## Introduction

This project explores how neural networks work. It provides tool to create, train, and test neural networks. 
It was created without relying on any external math or ML libraries such as NumPy, TensorFlow, or PyTorch.
## Features
- **Layers:** Build dense neural networks with any number of layers and parameters.
- **Activation functions:** Use various activation functions such as ReLU, Sigmoid, and Tanh.
- **Dropout:** Add custom dropout to improve model performance.
- **Loss:** Uses default Mean Squared Error (MSE) function.

## Application Demo

In the `application` section, a model is built to predict the sentiment of movie reviews based on the popular IMDB database. The model achieves 83% prediction accuracy. 

A pre-trained network is also available; `movie_reviews_sentiment_measure.py` runs an input window where a review can be pasted and evaluated by the model. It works best on actual IMDB reviews and longer texts, but it can also be used to evaluate any input.

## How to Use
### Creating a Network

Define the following to create a network:

- nodes list: The list size determines the number of layers, and list[i] determines the number of parameters at the given layer, ex:\
 &emsp; ```nodes = [50, 100, 10]``` creates a network with 3 layers having 50, 100, and 10 neurons respectively.
- dropout list: Defines the dropout rate at each layer except for the output layer. The dropout list size must equal the nodes list size minus one (no dropout on the output layer). If no parameter is used, no dropout is applied.
- activations list: Defines the activation function used at each layer except for the input layer. The activations list size must equal the nodes list size minus one (no activation on the input layer). If no parameter is used, no activation is applied.

#### Example

```network = Network([50, 100, 10], [0, 0.3], [sig, sig])```
This creates a 3-layer network with:
- Input size of 50
- Middle layer of 100 neurons with a 0.3 dropout rate
- Output size of 10
- Sigmoid activation on the middle and output layers

### Training the Network

Feed the `Network.learn` function with input/target list pairs. The input list size must match the number of parameters in the input layer of the network, and the target list size must match the number of parameters in the network output layer.
### Using the Trained Network

Apply the `Network.predict` function to the input. The input size must match the number of parameters in the network input layer.\
By default, the network uses mean squared error as the loss function and a 0.1 alpha learning rate.
