import numpy
from perceptron import Perceptron
from hopfield import SynchronousHopfieldNetwork
from backpropagation import BackpropagationNetwork
from lf import LogicFunction
from um import UpdateMethod


if __name__ == '__main__':
    # Algorytm uczenia perceptronu. Po wyświetleniu wykresu ze zmianami granicy decyzyjnej,
    # należy zamknąć wykres by kontynuować działanie programu.
    # Gdy uczenie perceptronu nie będzie w stanie znaleźć granicy decyzyjnej,
    # zostanie zwrócona para (None, iterations)

    print("")
    print("Zadanie 1 - algorytmy uczenia perceptronu")
    print("")

    perceptron_weights = [0.5, 0, 1]

    perceptron_and = Perceptron(LogicFunction.AND, perceptron_weights, 1.0, 1.0)
    weights, iterations = perceptron_and.train()
    print(f"Weights {weights} achieved after {iterations} iterations (simple perceptron training for AND function).")

    perceptron_and = Perceptron(LogicFunction.AND, perceptron_weights, 1.0, 1.0)
    weights, iterations = perceptron_and.bu_train()
    print(f"Weights {weights} achieved after {iterations} iterations "
          f"(batch update perceptron training for AND function).")

    perceptron_xor = Perceptron(LogicFunction.XOR, perceptron_weights, 1.0, 1.0)
    weights, iterations = perceptron_xor.bu_train()
    print(f"Weights {weights} achieved after {iterations} iterations "
          f"(batch update perceptron training for XOR function).")

    perceptron_xor = Perceptron(LogicFunction.XOR, perceptron_weights, 1.0, 1.0)
    weights, iterations = perceptron_xor.rbf_train()
    print(f"Weights {weights} achieved after {iterations} iterations "
          f"(batch update perceptron with RBF training for XOR function).")

    # Synchroniczna sieć Hopfielda

    print("")
    print("Zadanie 2 - synchroniczna sieć Hopfielda")
    print("")

    hopfield_points_3D = SynchronousHopfieldNetwork.generate_points(3, -1, 1)
    hopfield_weights_3D = 1/3 * numpy.array([
        [0, -2, 2],
        [-2, 0, -2],
        [2, -2, 0]
    ])
    control_signal_3D = [0, 0, 0]
    hopfield_network_3D = SynchronousHopfieldNetwork(3, hopfield_points_3D, hopfield_weights_3D, control_signal_3D)
    print("Macierz wag 3D - cechy: ")
    print(f"Jest symetryczna: {hopfield_network_3D.is_symmatrix()}")
    print(f"Ma zera na diagonali: {hopfield_network_3D.has_zeroes_on_diagonal()}")
    print(f"Jest dodatnio określona: {hopfield_network_3D.is_positive_definite()}")
    hopfield_results_3D = hopfield_network_3D.search_for_stability_points()
    print("Uzyskane wyniki:")
    SynchronousHopfieldNetwork.print_results(hopfield_results_3D)

    print("")

    hopfield_points_2D = SynchronousHopfieldNetwork.generate_points(2, -1, 1)
    hopfield_weights_2D = numpy.array([
        [0, 1],
        [-1, 0]
    ])
    control_signal_2D = [0, 0]
    hopfield_network_2D = SynchronousHopfieldNetwork(2, hopfield_points_2D, hopfield_weights_2D, control_signal_2D)
    print("Macierz wag 2D - cechy: ")
    print(f"Jest symetryczna: {hopfield_network_2D.is_symmatrix()}")
    print(f"Ma zera na diagonali: {hopfield_network_2D.has_zeroes_on_diagonal()}")
    print(f"Jest dodatnio określona: {hopfield_network_2D.is_positive_definite()}")
    hopfield_results_2D = hopfield_network_2D.search_for_stability_points()
    print("Uzyskane wyniki:")
    SynchronousHopfieldNetwork.print_results(hopfield_results_2D)

    print("")
    print("Zadanie 3 - propagacja wsteczna dla funkcji XOR")
    print("")

    backpropagation_weights = numpy.array([
        [0.86, -0.16, 0.28],
        [0.82, -0.51, -0.89],
        [0.04, -0.43, 0.48]
    ])
    backpropagation_network = BackpropagationNetwork(backpropagation_weights, 0.5)
    final_weights, iterations = backpropagation_network.train(UpdateMethod.TOTAL_ENERGY)
    print(f"Weights {final_weights} achieved after {iterations} iterations (Total Energy update method).")

    print("")

    backpropagation_network = BackpropagationNetwork(backpropagation_weights, 0.5)
    final_weights, iterations = backpropagation_network.train(UpdateMethod.PARTIAL_ENERGY)
    print(f"Weights {final_weights} achieved after {iterations} iterations (Partial Energy update method).")
