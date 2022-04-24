from enum import Enum


class LogicFunction(Enum):
    """
    Enum class that represents logic functions and their desired outputs.
    """
    AND = [0, 0, 0, 1]
    XOR = [0, 1, 1, 0]
