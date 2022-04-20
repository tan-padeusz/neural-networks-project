import array
import numpy
import matplotlib.pyplot as pyplot
from enum import Enum


class LogicFunction(Enum):
    """
    Enum class that represents logic functions and their desired outputs.
    """
    AND = [0, 0, 0, 1]
    XOR = [0, 1, 1, 0]


class Perceptron:
    """
    Class that represents perceptron network with single neuron.
    """
    class __PerceptronException(Exception):
        def __init__(self, message):
            super().__init__(message)

    @staticmethod
    def __validate_weights(weights):
        """
        Validates weight vector length.

        :param weights: Weights vector to be validated
        :return: If vector is valid, returns vector. Else raises PerceptronException.
        """
        if len(weights) != 3:
            raise Perceptron.__PerceptronException("Weight vector should have exactly 3 elements!")
        return weights

    @staticmethod
    def __validate_constants(constants):
        """
        Validates constant array (number of elements and their values).

        :param constants: Constant array to be validated.
        :return: If constant array is valid, returns constant array. Else raises PerceptronException.
        """
        if len(constants) != 2:
            raise Perceptron.__PerceptronException("Constants array should have exactly 2 elements!")
        for constant in constants:
            if constant == 0:
                raise Perceptron.__PerceptronException("No constant should be equal to 0!")
        return constants

    @staticmethod
    def __validate_stop_iteration(stop_iteration):
        """
        Validates stop iteration value.

        :param stop_iteration: Number of iterations after which training should be forcibly stopped.
        :return: If stop iteration value is valid, returns that value. Else raises PerceptronException.
        """
        if stop_iteration < 20:
            raise Perceptron.__PerceptronException("Perceptron should take at least 20 iterations to learn!")
        return stop_iteration

    def __init__(self, function: LogicFunction, weights: array, constants: array, stop_iteration: int = 200):
        """
        Perceptron constructor.

        :param function: Logic function that network should realise (AND or XOR).
        :param weights: 3-elemental array of weights used in training.
        :param constants: 2-elemental array of constants used in training. First element is "learn constant", second
                            element is "RBF constant". Both constants should not be 0!
        :param stop_iteration: Iteration after which training is forcibly stopped. Should be at least 20.
        """
        self.__train_inputs = [
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ]
        self.__train_outputs = function.value
        self.__weights = self.__validate_weights(weights)
        self.__constants = self.__validate_constants(constants)
        self.__stop_iteration = self.__validate_stop_iteration(stop_iteration)

    def __draw_inputs(self, figure=None):
        """
        Draws initial inputs.

        :param figure: Figure on which inputs will be drawn. If it's None, new figure will be given.
        :return: Figure on which inputs are drawn.
        """
        if figure is None:
            tmp, figure = pyplot.subplots()
        for train_input in self.__train_inputs:
            figure.scatter(train_input[1], train_input[2])
        return figure

    def __draw_function(self, start_x=0.0, end_x=1.0, label="", figure=None):
        """
        Draws linear function.

        :param start_x: Start x argument.
        :param end_x: End x argument.
        :param label: Function label.
        :param figure: Figure on which function will be drawn. If it's None, new figure will be given.
        :return: Figure on which function is drawn.
        """
        if figure is None:
            figure = pyplot.subplots()[1]
        y = self.__weights[2]

        # If y equals 0 then function is point.
        if y == 0:
            return figure
        # mlt = multiplier
        elif y > 0:
            mlt = -1
        else:
            mlt = 1

        x = numpy.linspace(start_x, end_x)
        y = mlt * (x * self.__weights[1] / y + self.__weights[0] / y)
        figure.plot(x, y, label=label, linewidth=1.5)
        return figure

    @staticmethod
    def __show_figure(figure):
        """
        Shows drawn inputs and functions with legend.

        :param figure: Figure on which inputs and functions were drawn.
        :return: Nothing.
        """
        figure.set_title("Inputs and decision-making boundaries.")
        figure.legend()
        pyplot.show()

    def __round_weights(self, round_digits):
        """
        Rounds found weights with specified precision.

        :param round_digits: Number of precision digits. If lower than 0, function does nothing.
        :return: Nothing.
        """
        if round_digits >= 0:
            rounded_weights = []
            for weight in self.__weights:
                rounded_weights.append(numpy.round(weight, round_digits))
            self.__weights = rounded_weights

    @staticmethod
    def __evaluate_output(dot_product):
        """
        Activation function for neuron.

        :param dot_product: Activation function input.
        :return: Classified output for neuron.
        """
        if dot_product > 0:
            return 1
        else:
            return 0

    @staticmethod
    def __dot_product(v1, v2):
        """
        Calculates dot product of two vectors.

        :param v1: First vector.
        :param v2: Second vector
        :return: Dot product of v1 and v2.
        """
        return float(numpy.dot(v1, v2))

    def __update_weights(self, cii, evaluated_output):
        """
        Updates weights during training with simple perceptron algorithm.

        :param cii: Current input index. Index of input currently tested.
        :param evaluated_output: Evaluated output for current input.
        :return: Nothing.
        """
        # im = input multiplier
        im = self.__constants[0] * (self.__train_outputs[cii] - evaluated_output)
        new_weights = []
        for index in range(3):
            new_weights.append(self.__weights[index] + im * self.__train_inputs[cii][index])
        self.__weights = new_weights

    def __update_bu_weights(self, bu_sum):
        """
        Updates weights during training with batch update perceptron algorithm.

        :param bu_sum: Sum used to determine next weights.
        :return: Nothing.
        """
        self.__weights = numpy.add(self.__weights, self.__constants[0] * bu_sum)

    def __evaluate_bu_sum(self, to_add):
        """
        Calculates sum used to determine next weights.

        :param to_add: Array of multipliers for summing inputs.
        :return: Sum of suitable vectors.
        """
        bu_sum = [0] * len(self.__train_inputs[0])
        for cii in range(len(self.__train_inputs)):
            # cii = current input index
            # tav = to add value
            tav = to_add[cii]
            if tav == 0:
                continue
            bu_sum = numpy.add(bu_sum, tav * numpy.array(self.__train_inputs[cii]))
        return bu_sum

    @staticmethod
    def __check_bu_end(to_add):
        """
        Checks if BUPA training should end.

        :param to_add: Array of multipliers for summing inputs.
        :return: True if training should end. False otherwise.
        """
        for value in to_add:
            if value != 0:
                return False
        return True

    def train(self, round_digits=-1):
        """
        Trains perceptron using simple perceptron algorithm.

        :param round_digits: Rounds weights with given precision. If lower than 0, does nothing. Default value is -1.
        :return: Pair of (weights, iterations taken) after training.
                    If weights were not found, returns pair of (None, None)
        """
        i = 0                          # i = iteration
        iwc = 0                        # iwc = iterations without change
        ic = len(self.__train_inputs)  # ic = input count

        figure = self.__draw_inputs()
        self.__draw_function(label="Iteration " + str(i), figure=figure)

        while iwc != ic:
            if i >= self.__stop_iteration:
                return None, None
            cii = i % ic               # cii = current input index
            dot_product = self.__dot_product(self.__train_inputs[cii], self.__weights)
            output: int = self.__evaluate_output(dot_product)

            if output == self.__train_outputs[cii]:
                iwc += 1
                if iwc != ic:
                    i += 1
            else:
                iwc = 0
                self.__update_weights(cii, output)
                i += 1
                self.__draw_function(label="Iteration " + str(i), figure=figure)

        self.__show_figure(figure)
        self.__round_weights(round_digits)
        return self.__weights, i + 1

    def bu_train(self, frt=False, round_digits=-1):
        """
        Trains perceptron using batch update perceptron algorithm.

        :param frt: Shows if this method was called from rbf_train method. Default value is False.
                    Making it "True" stops algorithm from drawing inputs and functions.
        :param round_digits: Rounds weights with given precision. If lower than 0, does nothing. Default value is -1.
        :return: Pair of (weights, iterations taken) after training.
                    If weights were not found, returns pair of (None, None)
        """
        i = 0                          # i = iteration
        ic = len(self.__train_inputs)  # ic = input count
        should_end = False

        if not frt:
            figure = self.__draw_inputs()
            self.__draw_function(label="Iteration " + str(i), figure=figure)

        while not should_end:
            if i >= self.__stop_iteration:
                return None, None
            to_add = [0] * ic

            # cii = current input index
            for cii in range(ic):
                dot_product = self.__dot_product(self.__train_inputs[cii], self.__weights)
                evaluated_output = self.__evaluate_output(dot_product)
                expected_output = self.__train_outputs[cii]
                to_add[cii] = expected_output - evaluated_output

            if self.__check_bu_end(to_add):
                should_end = True
            else:
                i += 1
                bu_sum = self.__evaluate_bu_sum(to_add)
                self.__update_bu_weights(bu_sum)
                if not frt:
                    # noinspection PyUnboundLocalVariable
                    self.__draw_function(label="Iteration " + str(i), figure=figure)

        if not frt:
            self.__show_figure(figure)
        self.__round_weights(round_digits)
        return self.__weights, i + 1

    # RBF kernel function
    def __rbf_kernel(self, x, y):
        numerator = numpy.power(-1 * numpy.absolute(x - y), 2)
        denominator = 2 * numpy.power(self.__constants[1], 2)
        return numpy.exp(numerator / denominator)

    # Trains perceptron with Batch Update
    # Perceptron Algorithm with RBF
    # kernel for solving XOR problem.
    # Returns a pair of array and int
    # array -> found weights
    # int -> iterations taken
    # Could return None if possible weights not found
    def rbf_train(self, round_digits=-1):
        new_inputs = []

        # cii = current input index
        for cii in range(len(self.__train_inputs)):
            train_input = self.__train_inputs[cii]
            rbf = self.__rbf_kernel(train_input[1], train_input[2])
            new_input: array = []
            for value in train_input:
                new_input.append(value)
            new_input.append(rbf)
            new_inputs.append(new_input)

        new_weights: array = []
        weight_rbf = self.__rbf_kernel(self.__weights[1], self.__weights[2])
        for value in self.__weights:
            new_weights.append(value)
        new_weights.append(weight_rbf)

        self.__train_inputs = new_inputs
        self.__weights = new_weights
        return self.bu_train(True, round_digits)
