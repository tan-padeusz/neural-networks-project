import numpy
from hopfield import SynchronousHopfieldNetwork


if __name__ == '__main__':
    points = SynchronousHopfieldNetwork.generate_points(3, -1, 1)
    weights = 1/3 * numpy.array([
        [0, -2, 2],
        [-2, 0, -2],
        [2, -2, 0]
    ])
    network = SynchronousHopfieldNetwork(3, points, weights, [0, 0, 0])
    SynchronousHopfieldNetwork.print_result(network.search_for_stability_points())

    print("")

    points = SynchronousHopfieldNetwork.generate_points(2, -1, 1)
    weights = numpy.array([
        [0, 1],
        [-1, 0]
    ])
    network = SynchronousHopfieldNetwork(2, points, weights, [0, 0])
    SynchronousHopfieldNetwork.print_result(network.search_for_stability_points())
