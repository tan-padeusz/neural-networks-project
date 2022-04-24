import numpy
from hopfield import SynchronousHopfieldNetwork
from backpropagation import BackpropagationNetwork


if __name__ == '__main__':
    # points = SynchronousHopfieldNetwork.generate_points(3, -1, 1)
    # weights = 1/3 * numpy.array([
    #     [0, -2, 2],
    #     [-2, 0, -2],
    #     [2, -2, 0]
    # ])
    # network = SynchronousHopfieldNetwork(3, points, weights, [0, 0, 0])
    # SynchronousHopfieldNetwork.print_results(network.search_for_stability_points())
    #
    # print("")
    #
    # points = SynchronousHopfieldNetwork.generate_points(2, -1, 1)
    # weights = numpy.array([
    #     [0, 1],
    #     [-1, 0]
    # ])
    # network = SynchronousHopfieldNetwork(2, points, weights, [0, 0])
    # SynchronousHopfieldNetwork.print_results(network.search_for_stability_points())

    weights = numpy.array([
        [0.86, -0.16, 0.28],
        [0.82, -0.51, -0.89],
        [0.04, -0.43, 0.48]
    ])
    network = BackpropagationNetwork(weights, 1, 1)
    final_weights, iteration = network.train()
    print("Weights ", final_weights, " achieved after ", iteration, " iterations!")
