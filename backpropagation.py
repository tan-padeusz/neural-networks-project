import numpy

from exception import BackpropagationException
from lf import LogicFunction


class BackpropagationNetwork:
    def __init__(self, initial_weights, alfa, learn_constant, stop_threshold=0.2):
        self.__train_inputs = [
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ]
        self.__train_outputs = LogicFunction.AND.value

        self.__weights = initial_weights
        self.__validate_initial_weights()

        self.__alfa = alfa
        self.__learn_constant = learn_constant
        self.__validate_constants()

        self.__stop_threshold = stop_threshold

    def __validate_initial_weights(self):
        if self.__weights.shape != (3, 3):
            raise BackpropagationException("Weight matrix must be square with 3 rows and 3 columns!")

    def __validate_constants(self):
        if self.__alfa == 0:
            raise BackpropagationException("Alfa must not be 0!")
        if self.__learn_constant == 0:
            raise BackpropagationException("Learn constant must not be 0!")

    @staticmethod
    def __activation_function(value):
        return 1.0 / (1.0 + numpy.exp(-value))

    @staticmethod
    def __activation_function_derivative(value):
        numerator = numpy.exp(-value)
        denominator = numpy.power(numpy.exp(-value) + 1, 2)
        return numerator / denominator

    def train(self):
        iteration = 0
        while True:
            iteration += 1
            gradients = []
            error = 0.0
            for index in range(3):
                train_input = self.__train_inputs[index]
                train_output = self.__train_outputs[index]
                output, gradient = self.__evaluate_gradient(train_input, train_output)
                gradients.append(gradient)
                error += (train_output - output) ** 2
            print("Error ", error, " at iteration ", iteration)
            if error <= self.__stop_threshold:
                return self.__weights, iteration
            if iteration >= 200:
                return self.__weights, iteration
            else:
                gradient_sum = numpy.sum(gradients)
                self.__weights = self.__weights - self.__alfa * gradient_sum

    def __evaluate_gradient(self, train_input, train_output):
        st_neuron_net = numpy.dot(train_input, self.__weights[0])
        st_neuron_out = self.__activation_function(st_neuron_net)

        nd_neuron_net = numpy.dot(train_input, self.__weights[1])
        nd_neuron_out = self.__activation_function(nd_neuron_net)

        mid_input = numpy.array([1, st_neuron_out, nd_neuron_out])
        rd_neuron_net = numpy.dot(mid_input, self.__weights[2])
        rd_neuron_out = self.__activation_function(rd_neuron_net)

        outputs = numpy.array([st_neuron_out, nd_neuron_out, rd_neuron_out])
        gradient = []
        ro = -2 * (train_output - outputs[2])
        derivative_value = self.__activation_function_derivative(outputs[2])

        for neuron in range(2):
            theta = derivative_value * self.__weights[2][neuron + 1]
            mid_derivative_value = self.__activation_function_derivative(outputs[neuron])
            delta_weights = numpy.zeros(3)
            for index in range(3):
                delta_weights[index] = ro * theta * mid_derivative_value * train_input[index]
            gradient.append(delta_weights)

        delta_weights = numpy.zeros(3)
        for index in range(3):
            delta_weights[index] = ro * derivative_value * mid_input[index]
        gradient.append(delta_weights)

        return rd_neuron_out, numpy.array(gradient)
