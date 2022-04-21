import numpy
from perceptron import Perceptron
from perceptron import LogicFunction
from hopfield import HopfieldNetwork


if __name__ == '__main__':
    # inputs = [
    #     [1, 0, 0],
    #     [1, 0, 1],
    #     [1, 1, 0],
    #     [1, 1, 1]
    # ]
    # outputs = [
    #     0,
    #     1,
    #     1,
    #     0
    # ]
    # perceptron = Perceptron(inputs, outputs)
    # weights, iterations = perceptron.rbf_bu_train()
    # print("Weights: " + str(weights))
    # print("Iterations: " + str(iterations))

    # weight_matrix = 1/3 * numpy.matrix([
    #     [0, -2, 2],
    #     [-2, 0, -2],
    #     [2, -2, 0]
    # ])
    # hopfield_inputs = [
    #     [1, -1, 1],
    #     [-1, 1, -1],
    #     [1, 1, 1]
    # ]
    # control_signal = [
    #     0,
    #     0,
    #     0
    # ]
    # network = HopfieldNetwork(weight_matrix, hopfield_inputs, control_signal)
    # for result in network.check_stability_points():
    #     print("Point ", result[0], " has stabilised at ", result[1], ". Iterations taken: ", result[2])
    points = HopfieldNetwork.generate_points(3, -1)
    weights = 1/3 * numpy.matrix([
        [0, -2, 2],
        [-2, 0, -2],
        [2, -2, 0]
    ])
    # minor change
    control_signal = [0, 0, 0]
    hn = HopfieldNetwork(3, points, weights, control_signal)
    for result in hn.check_stability_points():
        print("Point ", result[0], " has stabilised at ", result[1], ". Iterations taken: ", result[2])
    # perceptron = Perceptron(LogicFunction.XOR, [0.5, 0, 1], [1, 1])
    # result = perceptron.rbf_train(3)
    # print("Result weights: ", result[0])
    # print("Iterations: ", result[1])
