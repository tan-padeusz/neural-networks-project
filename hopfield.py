import numpy.linalg
from exception import HopfieldNetworkException


class SynchronousHopfieldNetwork:
    def __init__(self, neuron_count, points, weights, control_signal, stop_iteration=1000):
        """
        Class that represents Hopfield network that works synchronously.

        :param neuron_count: Number of neurons used in network.
        :param points: Points that network works on.
        :param weights: Weights that network works on.
        :param control_signal: Control signal that network works on.
        :param stop_iteration: Iteration after which network is stopped.
        """
        self.__neuron_count = neuron_count
        self.__validate_neuron_count()

        self.__points = points
        self.__validate_points()

        self.__weights = weights
        self.__validate_weights()

        self.__control_signal = control_signal
        self.__validate_control_signal()

        self.__stop_iteration = stop_iteration
        self.__validate_stop_iteration()

    def __validate_neuron_count(self):
        """
        Validates "neuron_count" property.

        :return: Nothing.
        :raise HopfieldNetworkException: If neuron count is lesser than 2.
        """
        if self.__neuron_count < 2:
            raise HopfieldNetworkException("Network must have at least 2 neurons to work properly!")

    def __validate_points(self):
        """
        Validates "points" property.

        :return: Nothing.
        :raise HopfieldNetworkException: If number of points is 0 or
                                         if any of points length is not equal to neuron count.
        """
        if len(self.__points) == 0:
            raise HopfieldNetworkException("Network must have at least one point to work on!")
        for point in self.__points:
            if len(point) != self.__neuron_count:
                raise HopfieldNetworkException("All points must have length equal to neuron count!")

    def __validate_weights(self):
        """
        Validates "weights" property.

        :return: Nothing.
        :raise HopfieldNetworkException: If matrix is not square with number of rows and columns equals to neuron count.
        """
        if self.__weights.shape != (self.__neuron_count, self.__neuron_count):
            raise HopfieldNetworkException("Weights matrix should be square with number of rows and columns"
                                           " equals to " + str(self.__neuron_count) + "!")

    def __validate_control_signal(self):
        """
        Validates "control_signal" property.

        :return: Nothing.
        :raise HopfieldNetworkException: If control signal array length is not equal to neuron count.
        """
        if len(self.__control_signal) != self.__neuron_count:
            raise HopfieldNetworkException("Control signal element count should be equal to neuron count!")

    def __validate_stop_iteration(self):
        """
        Validates "stop_iteration" property.

        :return: Nothing.
        :raise HopfieldNetworkException: If stop iteration value is lower than 1000.
        """
        if self.__stop_iteration < 1000:
            raise HopfieldNetworkException("Network should take at least 1000 iterations before "
                                           "being forcibly stopped!")

    def is_symmatrix(self):
        """
        Checks if weight matrix is symmetric.

        :return: True if weight matrix is symmetric. Else returns false.
        """
        return numpy.allclose(self.__weights, numpy.transpose(self.__weights))

    def has_zeroes_on_diagonal(self):
        """
        Checks if diagonal of weight matrix consists of zeroes.

        :return: True if all elements on diagonal are zeroes. Else returns False.
        """
        diagonal = numpy.diagonal(self.__weights)
        return numpy.allclose(diagonal, 0)

    def is_positive_definite(self):
        """
        Checks if weight matrix is positive definite.

        :return: True if weight matrix is positive definite. Else returns False.
        """
        return self.is_symmatrix() and numpy.all(numpy.linalg.eigvals(self.__weights) > 0)

    def search_for_stability_points(self):
        """
        Searches points for stability points or cycles.

        :return: List of triples of (initial_point, has_stabilised, result).
        """
        result = []
        for point in self.__points:
            result.append(self.__search_for_stability_points(point))
        return result

    def __search_for_stability_points(self, point):
        """
        Checks if given point is stability point or enters cycle.

        :param point: Point to be checked.
        :return: Triple of (initial_point, has_stabilised, result)
        """
        iteration = 0
        neurons_without_change = 0
        old_point = numpy.copy(point)
        history = [old_point]

        while iteration < self.__stop_iteration:
            iteration += 1
            u_vector = (self.__weights.dot(old_point) + self.__control_signal)
            new_point = numpy.zeros(self.__neuron_count)
            for index in range(self.__neuron_count):
                new_point[index] = self.__activation_function(u_vector[index])
            history.append(new_point)
            historical_index = self.__search_history_for_point(history, new_point)
            if historical_index > -1:
                return [point, False, self.__get_history_fragment(history, historical_index, iteration)]
            if numpy.array_equal(new_point, old_point):
                neurons_without_change += 1
                if neurons_without_change == self.__neuron_count + 1:
                    return [point, True, new_point.astype(int)]
            else:
                neurons_without_change = 0
                old_point = new_point
        return [None, None, None]

    @staticmethod
    def __activation_function(value):
        """
        Activation function used by network.

        :param value: Activation function parameter.
        :return: 1 if value is greater than 0, else returns -1.
        """
        if value > 0:
            return 1
        else:
            return -1

    @staticmethod
    def __search_history_for_point(history, point):
        """
        Browses network history to find point that is equal to point given as parameter. Between found point and current
        point must be at least one different point.

        :param history: History to be searched.
        :param point: Point to be found in history.
        :return: Point index if history contains searched point and between it and current point there is at least one
                 different point. Else returns -1.
        """
        found_between = False  # indicates if between two equal points was another, different point
        for iteration in range(len(history) - 1, -1, -1):
            equal = numpy.array_equal(history[iteration], point)
            if not equal:
                found_between = True
            elif found_between:
                return iteration
        return -1

    @staticmethod
    def __get_history_fragment(history, start_index, end_index):
        """
        Returns fragment of history between given indexes (inclusive).

        :param history: History from which fragment is taken.
        :param start_index: Index of first element of history to take (inclusive).
        :param end_index: Index of last element of history to take (inclusive).
        :return:
        """
        result = []
        for index in range(start_index, end_index + 1):
            result.append(history[index].astype(int).tolist())
        return result

    @staticmethod
    def generate_points(size, lower_value=0, upper_value=1):
        """
        Generates points to be used in Hopfield network.

        :param size: Size of point.
        :param lower_value: Binary 0 will be converted into that value.
        :param upper_value: Binary 1 will be converted into that value.
        :return: Array of points to be used in Hopfield network.
        """
        max_value = numpy.power(2, size)
        points = []
        for value in range(max_value):
            value_array = []
            string_value = format(value, 'b')
            while len(string_value) < size:
                string_value = "0" + string_value
            for char in string_value:
                if char == '0':
                    value_array.append(lower_value)
                else:
                    value_array.append(upper_value)
            points.append(value_array)
        return points

    @staticmethod
    def print_result(results):
        """
        Prints Hopfield network results.

        :param results: Results to be printed.
        :return: Nothing.
        """
        for (initial_point, has_stabilised, result) in results:
            message = "Point " + str(initial_point) + " has "
            if has_stabilised:
                message += "stabilised at " + str(result) + "."
            else:
                message += "entered cycle: " + str(result) + "."
            print(message)
