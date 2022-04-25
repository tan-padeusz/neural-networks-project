class PerceptronException(Exception):
    """
    Exception used in perceptron training.
    """
    def __init__(self, message):
        super().__init__(message)


class HopfieldNetworkException(Exception):
    """
    Exception used in Hopfield network.
    """
    def __init__(self, message):
        super().__init__(message)


class BackpropagationException(Exception):
    """
    Exception used in bacpropagation training.
    """
    def __init__(self, message):
        super().__init__(message)
