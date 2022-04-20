import numpy.linalg


class HopfieldException(Exception):
    def __init__(self, message):
        super().__init__(message)


class HopfieldNetwork:
    # Validates initial data
    def __validate_data(self):
        if len(self.__points) == 0:
            raise HopfieldException("Network needs at least one point to learn!")
        for points in self.__points:
            if len(points) != self.__neuron_count:
                raise HopfieldException("All points should have " + str(self.__neuron_count) + " elements!")
        if self.__weights.shape != (self.__neuron_count, self.__neuron_count):
            raise HopfieldException("Weight matrix should be square with " + str(self.__neuron_count) + " rows"
                                    + " and " + str(self.__neuron_count) + " columns!")

    # Checks if weight matrix is symmetric.
    def __check_symmatrix(self):
        return numpy.transpose(self.__weights) == self.__weights

    # Checks if elements on weight matrix's diagonal are zeroes.
    def __check_diagonal_zeroes(self):
        diagonal = numpy.diagonal(self.__weights)
        for value in diagonal:
            if value != 0:
                return False
        return True

    # Checks if weight matrix is positive definite.
    def __check_positive_definite(self):
        copy_matrix = self.__weights
        while copy_matrix.shape != (1, 1):
            if numpy.linalg.det(copy_matrix) <= 0:
                return False
            copy_matrix = numpy.delete(copy_matrix, [0, 0])
        if copy_matrix[0] <= 0:
            return False
        else:
            return True

    # Activation function
    @staticmethod
    def __evaluate_output(value):
        if value > 0:
            return 1
        else:
            return -1

    # Hopfield network constructor
    def __init__(self, neuron_count, points, weights, control_signal):
        self.__neuron_count = neuron_count
        self.__points = points
        self.__weights = weights
        self.__control_signal = control_signal
        self.__validate_data()

    # Checks all points for stability points.
    def check_stability_points(self):
        results = []
        for point in self.__points:
            results.append(self.__check_stability_points(point))
        return results

    # Checks single point for stability point.
    def __check_stability_points(self, point):
        i = -1    # iteration
        n = 0    # current neuron
        nwc = 0  # neurons without change
        copy_point = numpy.copy(point)
        while nwc != self.__neuron_count + 1:
            i += 1
            # uv = u value
            uv = (self.__weights.dot(copy_point) + self.__control_signal)[0, n]
            output = self.__evaluate_output(uv)
            if output != copy_point[n]:
                nwc = 0
                copy_point[n] = output
            else:
                nwc += 1
            n = (n + 1) % self.__neuron_count
        return [point, copy_point, i]

    @staticmethod
    def __check_history(history, point, neuron):
        for (h_point, h_neuron, h_iteration) in history:
            if numpy.array_equal(h_point, point) and h_neuron == neuron:
                return h_iteration
        return -1

    # Checks single point for stability point or cycle.
    def __check_cycle(self, point):
        i = -1   # iteration
        n = 0    # current neuron
        nwc = 0  # neurons without change
        copy_point = numpy.copy(point)
        history = []
        while nwc != self.__neuron_count + 1:
            i += 1
            # uv = u value
            uv = (self.__weights.dot(copy_point) + self.__control_signal)[0, n]
            output = self.__evaluate_output(uv)
            if output != copy_point[n]:
                nwc = 0
                copy_point[n] = output
            else:
                nwc += 1
            # fih = found in history
            fih = self.__check_history(history, copy_point, n)
            if fih != -1:
                return [point, i - fih, i]
            history.append((copy_point, n, i))
            n = (n + 1) % self.__neuron_count
        return [point, copy_point, i]

    # Generates points to be used in network.
    @staticmethod
    def generate_points(size, lower_value=0, upper_value=1):
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
