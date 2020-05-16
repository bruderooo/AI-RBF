from Include.help_functions import euclidean_distance
from matplotlib import pyplot as plt
import numpy as np


def gauss_function(r: np.ndarray, sigma: np.float):
    return np.exp(-(r ** 2) / (2 * sigma ** 2))


if __name__ == '__main__':
    # H = int(input("Podaj ilosc neuronow ukrytych"))
    H = 8

    eta = 0.01

    data = np.loadtxt("./Data/approximation_train_1.txt")
    data_matrix = np.asarray([
        np.asarray([
            euclidean_distance(datumA, datumB)
            for indexB, datumB in enumerate(data)
        ])
        for indexA, datumA in enumerate(data)
    ])

    layer1_weighs = np.dot(np.linalg.pinv(data_matrix), data[1])
    layer2_weighs = 2 * np.random.rand(H) - 1

    epochs = 10_000

    for epoch in range(epochs):
        z = np.dot(gauss_function(data_matrix, 0.5), layer1_weighs)
        y = np.dot(layer2_weighs, z)

        # Obliczanie gradientu dla warstwy ostatniej
        dE_dw2 = (y - data[1]) * y * z

        layer2_weighs -= dE_dw2 * eta
