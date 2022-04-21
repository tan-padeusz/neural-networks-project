import numpy
from hopfield import SynchronousHopfieldNetwork


if __name__ == '__main__':
    points = SynchronousHopfieldNetwork.generate_points(3, -1, 1)
    weights = numpy.array([
        [0, -2, 2],
        [-2, 0, -2],
        [2, 0, -1]
    ])
    network = SynchronousHopfieldNetwork(3, points, weights, [0, 0, 0])
    for result in network.search_for_stability_points():
        print(result)
