'''
#@Author: Grizlly 
# @Date: 2018-05-03 15:57:55 
# -*- coding: utf-8 -*- 
'''
from numpy import exp, array, random, dot


# NeuralNetwork Build
class NeuralNetwork():
    def __init__(self):
        # random seeds for get same result every time
        random.seed(1)

        # single neural builded, including 3 input-connects and 1 output-connect
        # giving random weights to the matrix 3 x 1, the scope of -1 ~1. the  average is 0
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # Sigmoid function, the ogee
    # Use this function to normalize the weighted sum of the input, making it range from 0 to 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

        # The derivative of the Sigmoid function.
        # The gradient of the Sigmoid curve.

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs,
              number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # import training neuralwork
            output = self.think(training_set_inputs)

            # calculate error
            error = training_set_outputs - output

            adjustment = dot(training_set_inputs.T,
                             error * self.__sigmoid_derivative(output))

            # adjust weights
            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == '__main__':
    # init nerualwork
    neural_network = NeuralNetwork()

    print("random initial weight ")
    print(neural_network.synaptic_weights)

    # the training set. four samples
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # train the neuralwork
    neural_network.train(training_set_inputs, training_set_outputs, 10000)
    print("Post-workout synaptic weight.")
    print(neural_network.synaptic_weights)

    # use new data to test the neuralWork
    print("[1,0,0] -> ?:")
    print(neural_network.think(array([1, 0, 0])))
