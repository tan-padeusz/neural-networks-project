from enum import Enum


class LogicFunction(Enum):
    """
    Enum class that represents logic functions and their desired outputs.
    """
    AND = [0, 0, 0, 1]
    XOR = [0, 1, 1, 0]


class UpdateMethod(Enum):
    """
    Enum class that represents update method used in backpropagation.
    """
    PARTIAL_ENERGY = -1
    TOTAL_ENERGY = 1
