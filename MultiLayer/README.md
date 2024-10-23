# Multi-Layer Network
Multi-layer neural network with back propagation to recognize a single digit from an image.

## Problem description
Single handwritten digit recognition from an image.

We have 28x28 pixels images of handwritten digits. This is a classification problem so each number represents one class. The network should classify the image to its value (number).

Images were taken from Mnist handwritten digits dataset.

## Network function
Multi-layer network we can generally split into tree parts.
1. First part is an input layer. This layer is typically linear which means that it only passes the input values. Number of neurons in this layer is set by number of input values.
2. Input layer is followed by hidden layers. Generally there can be any number of hidden layers with anu number of neurons.
3. Last layer is an output layer. Number of neurons in this layer is set by application (number of outputs is based on the problem). Outputs of these neurons are also outputs of the network.

<img src="README_img/network.png" alt="Network scheme" title="Network scheme" width=50%>

*Network scheme*

Each neuron is specified by weights, bias and activation function.

Network with back propagation separates the classes with lines as boundaries (this comes from the way of calculating z). Multilayer network can also separate nonlinear input.

Property of this type of network is that every input is somehow classified. This means than unknown or even a wrong input is always assigned to some output. Wrong input is some input for which the network isn't trained (for example network is trained to recognize digits and we input a letter).

<img src="README_img/separation.png" alt="Illustration of a possible multi-layer network separation" title="Illustration of a possible multi-layer layer network separation" width=30%>

*Illustration of a possible multi-layer layer network separation*

## Network training
Training in this case was with back propagation method with parameters adjusting after each training sample. This approach isn't optimal for a dataset of this size (60 000 training samples). So in the example not whole dataset was used.

Possible solution could be to process the input data in baches and adjust the parameters after each bach not sample.

### Back propagation
Error back propagation is an iterative gradient algorithm that minimizes the square of error function. Practically that means that error propagating from output back to input. This backwards propagation helps adjust weights and biases. Overall goal is to find the minimum of networks loss (error) function. This method is used with supervised learning (because we need to be able to calculate error).

### Training

The training process then can be split into tree steps.
1. Forward pass: training data are input into the network and outputs of each neuron is calculated. At the end there are predictions fo the output layer.
2. Backward pass: we count the error of the output (based on labels of training data). This error is then propagated from the output layer back to the input layer. On the way errors for each neuron is counted.
3. Adjusting: based on neurons outputs and their errors adjusting of the weights and biases is done.

In this example in addition to learning rate which defines the adjusting steps a momentum is also defined. With momentum we determine how much the weights or biases are changing in each iteration. The bigger changes the bigger adjusting steps. Momentum should help network to converge faster (converge means that the network is trained).

<img src="README_img/error_fnc.png" alt="Error function without momentum" title="Error function without momentum" width=50%>

*Error function without momentum*

<img src="README_img/error_fnc_momentum.png" alt="Error function with momentum" title="Error function with momentum" width=50%>

*Error function with momentum*

*Note: in both cases training was done with same learning rate and on average learning with momentum converge twice as fast.*

### Formulas

Output of a single neuron:

$y = f(z)$

$z = \sum_{i}(w_{i} * x_{i}) + b$

*f(z) is an activation function*

Errors:

$E_{sample} = \frac{1}{2} \sum_i(d_i - y_i)^2$

*labels - outputs*

$E_{network} = \sum_i(E_{sample_i})$

Back propagation:

$\delta = (d - y) * \frac{df(z)}{dz}$

*For output layer (labels - outputs)*

$\delta = \sum_i(w_i * (d_i - y_i)) * \frac{df(z)}{dz}$

*For hidden layers (error from previous layer * weights of previous layer, starting at the output layer)*

Adjusting:

$w_{i+1} = w_i + rate * \delta * input$

$b_{i+1} = b_i + rate * \delta$

*rate is learning rate and input is input of that layer (neuron) not network input (except the first layer)*

$w_{i+1} = w_i + rate * \delta * input + momentum * (w_i - w_{i-1})$

$b_{i+1} = b_i + rate * \delta * input + momentum * (b_i - b_{i-1})$

*With momentum*


## Implementation

### Topology
Because we have 28x28 input images the input layer contains 784 (28 * 28) input values. Then there are 2 hidden layers. First one has 40 neurons and second has 20. Output layer has 10 neurons (for digits 0-10).

### Activation function
As activation function hyperbolic tangent (tanh) function was used.

<img src="README_img/tanh.png" alt="Hyperbolic tangent function" title="Hyperbolic tangent function" width=50%>

*Hyperbolic tangent function*

$f(x) = \frac{1 - e^{-x}}{1 + e^{-x}}$

$f(x)' = 1 - f(x)^2$

### Model functionality
Weights and biases are randomly generated for each network object.

When training: training data, training labels, learning rate and acceptable error must be provided. Optionally momentum, epochs and if a network error development graph should be shown. Training will end either when the network error is lower than the acceptable error or when training process reaches max iterations (set by epochs parameter.)

After training network testing can be done. For testing we need to provide testing data with testing labels (labels as output value not matrix of neurons activations). Result of the testing is accuracy optionally with number of training samples, number of errors and dictionary with errors frequencies.


### Training and Testing
Training was done with 1000 samples for acceptable error 0.1 (10%). For training 100 samples was used. When testing average accuracy was about 20%.

<img src="README_img/error_freq.png" title="Errors frequencies" alt="Error frequencies" width=50%>

*Error frequencies for 100 testing samples (accuracy 0.86)*

<img src="README_img/single_output.png" title="Single image prediction" alt="Single image prediction" width=50%>

*Single image prediction (network assigned value 2 with probability 0.996)*

## Notes
- When i was using sigmoid function as an activation function the network didn't converge.
- Momentum really speed up the training process.
- Learning rate, momentum, initial weight and initial biases - each of these has noticeable influence on network convergence.
- The biggest challenge of this project were mathematics behind the back propagation and tracking all the values in matrixes. To look for right dimensions and right operators.
- Training with huge amount of training samples is really slow (for these cases training with batches could be better).
- When using tanh function labels -1/1 are more optional than 0/1.