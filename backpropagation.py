import numpy
import matplotlib.pyplot as pyplot

from enums import LogicFunction, UpdateMethod
from exceptions import BackpropagationException


class BackpropagationNetwork:
    def __init__(self, initial_weights, learn_constant, stop_threshold=0.2, stop_iteration=3):
        """
        Class that represents backpropagation network.

        :param initial_weights: Initial weights used in network.
                                Must be matrix with 3 rows and 3 columns.
        :param learn_constant: Learn constant used in training.
        :param stop_threshold: If evaluated error is lesser than this value, training stops.
        :param stop_iteration: If iteration count is at least this value, training stops.
        """
        self.__train_inputs = [
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ]
        self.__train_outputs = LogicFunction.XOR.value

        self.__weights = initial_weights
        self.__validate_initial_weights()

        self.__learn_constant = learn_constant
        self.__validate_learn_constant()

        self.__stop_threshold = stop_threshold
        self.__validate_stop_threshold()

        self.__stop_iteration = stop_iteration
        self.__validate_stop_iteration()

    def __validate_initial_weights(self):
        """
        Validates "weights" property.

        :raise BackpropagationException: If weight matrix is not square matrix with 3 rows and 3 columns.
        """
        if self.__weights.shape != (3, 3):
            raise BackpropagationException("Weight matrix must be square with 3 rows and 3 columns!")

    def __validate_learn_constant(self):
        """
        Validates "learn constant" property.

        :raise BackpropagationException: If property value is equal to 0.
        """
        if self.__learn_constant == 0:
            raise BackpropagationException("Learn constant must not be 0!")

    def __validate_stop_threshold(self):
        """
        Validates "stop_threshold" property.

        :raise BackpropagationException: If property value is not greater than 0.
        """
        if self.__stop_threshold <= 0.0:
            raise BackpropagationException("Stop threshold must be greater than 0!")

    def __validate_stop_iteration(self):
        """
        Validates "stop iteration" property.

        :raise BackpropagationException: If property value is lower than 5.
        """
        if self.__stop_iteration < 3:
            raise BackpropagationException("There must be at least 5 iterations to forcibly stop training!")

    @staticmethod
    def __activation_function(value):
        """
        Activation function for neurons.

        :param value: Parameter of activation function.
        :return: Output of activation function.
        """
        return 1.0 / (1.0 + numpy.exp(-value))

    @staticmethod
    def __activation_function_derivative(value):
        """
        Derivative of activation function for neurons.

        :param value: Parameter of derivative of activation function.
        :return: Output of derivative of activation function.
        """
        numerator = numpy.exp(-value)
        denominator = numpy.power(numpy.exp(-value) + 1, 2)
        return numerator / denominator

    def __update_weights(self, value):
        """
        Updates network weights.

        :param value: Value by which weights will be updated (multiplied by learn constant).
        """
        self.__weights -= self.__learn_constant * numpy.array(value)

    @staticmethod
    def __sum_gradients(gradients):
        """
        Calculates sum of evaluated gradients (matrices addition).

        :parameter gradients: An array of evaluated gradients.
        :returns: Sum of gradients in array,
        """
        result = numpy.zeros((3, 3))
        for row in range(3):
            for column in range(3):
                for gradient in range(4):
                    result[row, column] += gradients[gradient][row][column]
        return result

    def train(self, update_method: UpdateMethod):
        """
        Trains network.

        :param update_method: Parameter that indicates when weights should be updated.
        :returns: A pair of (final_weights, iteration_count).
        """
        pyplot.close()
        iteration = 0
        figure = None
        while True:
            iteration += 1
            gradients = []
            error = 0.0
            for index in range(4):
                train_input = self.__train_inputs[index]
                train_output = self.__train_outputs[index]
                output, gradient = self.__evaluate_gradient(train_input, train_output)
                gradients.append(gradient)
                energy = numpy.power(train_output - output, 2)
                figure = self.__draw_energy((iteration - 1) * 4 + index, energy, figure)
                error += energy

                if update_method == UpdateMethod.PARTIAL_ENERGY:
                    self.__update_weights(gradient)

            if error <= self.__stop_threshold or iteration >= self.__stop_iteration:
                self.__show_energy_change(figure, update_method)
                return self.__weights, iteration

            if update_method == UpdateMethod.TOTAL_ENERGY:
                self.__update_weights(self.__sum_gradients(gradients))

    def __evaluate_gradient(self, train_input, train_output):
        """
        Evaluates gradient for network.

        :param train_input: Input used in network.
        :param train_output: Desired output.
        :returns: Network gradient.
        """
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

        return rd_neuron_out, gradient

    @staticmethod
    def __draw_energy(iteration, energy, figure=None):
        """
        Draws energy point on figure.

        :param iteration: X-value on diagram.
        :param energy: Y-value on diagram.
        :param figure: Figure on which diagram is drawn.
                       If it's None, new figure will be given,
        :returns: Figure on which diagram is drawn.
        """
        if figure is None:
            tmp, figure = pyplot.subplots()
        figure.scatter(iteration, energy)
        return figure

    @staticmethod
    def __show_energy_change(figure, mode):
        """
        Shows energy change diagram.

        :param figure: Figure on which diagram is drawn.
        :param mode: Weights update mode.
        """
        figure.set_title(f"Energy change - {mode.value}")
        figure.grid()
        pyplot.xlabel("Iteration")
        pyplot.ylabel("Energy")
        pyplot.show()
