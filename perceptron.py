import numpy
import matplotlib.pyplot as pyplot

from enums import LogicFunction
from exceptions import PerceptronException


class Perceptron:
    """
    Class that represents perceptron network with single neuron.
    """
    def __init__(self, function: LogicFunction, weights, learn_constant, rbf_constant, stop_iteration=200):
        """
        Perceptron constructor.

        :param function: Logic function that network should realise (AND or XOR).
        :param weights: 3-elemental array of weights used in training.
        :param learn_constant: Learn constant used in training.
        :param rbf_constant: RBF constant used in RBF training.
        :param stop_iteration: Iteration after which training is forcibly stopped. Should be at least 20.
        """
        self.__train_inputs = [
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ]
        self.__train_outputs = function.value

        self.__weights = weights
        self.__validate_weights()

        self.__learn_constant = learn_constant
        self.__rbf_constant = rbf_constant
        self.__validate_constants()

        self.__stop_iteration = stop_iteration
        self.__validate_stop_iteration()

    def __validate_weights(self):
        """
        Validates "weights" property.

        :raises PerceptronException: If property length is not equal to 3.
        """
        if len(self.__weights) != 3:
            raise PerceptronException("Weight vector should have exactly 3 elements!")

    def __validate_constants(self):
        """
        Validates constants used in training.

        :raises PerceptronException: If "learn_constant" or "rbf_constant" equals 0.
        """
        if self.__learn_constant == 0:
            raise PerceptronException("Learn constant should not be equal to 0!")
        if self.__rbf_constant == 0:
            raise PerceptronException("RBF constant should not be equal to 0!")

    def __validate_stop_iteration(self):
        """
        Validates "stop_iteration" property.

        :raises PerceptronException: If property value is lesser than 20.
        """
        if self.__stop_iteration < 20:
            raise PerceptronException("Perceptron should take at least 20 iterations to learn!")

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
        im = self.__learn_constant * (self.__train_outputs[cii] - evaluated_output)
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
        self.__weights = numpy.add(self.__weights, self.__learn_constant * bu_sum)

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
                    If weights were not found, returns pair of (None, iterations taken)
        """
        i = 0                          # i = iteration
        iwc = 0                        # iwc = iterations without change
        ic = len(self.__train_inputs)  # ic = input count

        figure = self.__draw_inputs()
        self.__draw_function(label="Iteration " + str(i), figure=figure)

        while iwc != ic:
            if i >= self.__stop_iteration:
                return None, i + 1
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

    def bu_train(self, round_digits=-1):
        """
        Trains perceptron using batch update perceptron algorithm.

        :param round_digits: Rounds final weights with given precision. If lower than 0, does nothing.
                             Default value is -1.
        :return: Pair of (weights, iterations taken) after training.
                 If weights were not found, returns pair of (None, iterations taken).
        """
        return self.__bu_train(False, round_digits)

    def __bu_train(self, frt, round_digits=-1):
        """
        Trains perceptron using batch update perceptron algorithm.

        :param round_digits: Rounds weights with given precision.
                             If lower than 0, does nothing.
                             Default value is -1.
        :param frt: Shows if this method was called from rbf_train method.
                    If value is "True", prevents bu_train from drawing inputs and decision boundaries.
        :return: Pair of (weights, iterations taken) after training.
                 If weights were not found, returns pair of (None, iterations taken)
        """
        i = 0                          # i = iteration
        ic = len(self.__train_inputs)  # ic = input count
        should_end = False

        if not frt:
            figure = self.__draw_inputs()
            self.__draw_function(label="Iteration " + str(i), figure=figure)

        while not should_end:
            if i >= self.__stop_iteration:
                return None, i
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

    def __rbf_kernel(self, x, y):
        """
        Calculates RBF kernel function for given x and y.

        :param x: x value
        :param y: y value
        :return: Evaluated kernel output for given values.
        """

        numerator = numpy.power(-1 * numpy.absolute(x - y), 2)
        denominator = 2 * numpy.power(self.__rbf_constant, 2)
        return numpy.exp(numerator / denominator)

    def rbf_train(self, round_digits=-1):
        """
        Trains perceptron using batch update perceptron algorithm
        with expanding dimensions of inputs and weights using RBF kernel.

        :param round_digits: Rounds weights with given precision. If lower than 0, does nothing. Default value is -1.
        :return: Pair of (weights, iterations taken) after training.
                    If weights were not found, returns a pair of (None, iterations taken)
        """
        for train_input in self.__train_inputs:
            rbf = self.__rbf_kernel(train_input[1], train_input[2])
            train_input.append(rbf)

        weight_rbf = self.__rbf_kernel(self.__weights[1], self.__weights[2])
        self.__weights.append(weight_rbf)
        return self.__bu_train(True, round_digits)
