__author__ = "Szymon Jacoń"

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from Include.help_functions import make_matrix, gauss_function, euclidean_distance


def how_much_var_in_array(array, var):
    n = 0
    for i in array:
        if i[-1] == var:
            n += 1
    return n


def take_some_lines_from_vector(number_of_lines, vector):
    a = np.arange(np.shape(vector)[0])
    np.random.shuffle(a)

    tmp_vector = np.asarray([vector[z] for z in a])

    return tmp_vector[:number_of_lines]


def euclidean_distance_point_to_vector(point, basis):
    tmp_list = []
    for vector in basis:
        tmp_list += [euclidean_distance(point, vector)]

    return tmp_list


def choose_centers_with_k_means_clustering(data_to_be_grouped, number_of_groups):
    clusters = take_some_lines_from_vector(number_of_groups, data_to_be_grouped)
    M = len(data_to_be_grouped)
    data_to_be_grouped = np.concatenate((data_to_be_grouped, np.zeros((M, 1))), axis=1)
    max_distance = 0.0

    for cycle in range(10):
        tmp_clusters = np.zeros((number_of_groups, data_to_be_grouped.shape[1] - 1))

        for vector_index, vector in enumerate(data_to_be_grouped):
            distance = np.asarray(euclidean_distance_point_to_vector(vector[:-1], clusters))
            sorted_distance = np.sort(distance)
            max_distance = sorted_distance[-1]
            index_cluster = np.where(distance == sorted_distance[0])[0][0]

            vector[-1] = index_cluster

            for which_coordinate, coordinate in enumerate(vector[:-1]):
                tmp_clusters[index_cluster][which_coordinate] += coordinate

        for i in range(number_of_groups):
            tmp = how_much_var_in_array(data_to_be_grouped, i)
            if 0 != tmp:
                for which_cluster_coordinate, cluster_coordinate in enumerate(tmp_clusters[i]):
                    clusters[i][which_cluster_coordinate] = cluster_coordinate / tmp

    return clusters, max_distance


if __name__ == '__main__':

    if len(sys.argv) == 1:
        choice = input(
            "Choose 1 if you want make classification, or choose 2 if u want make approximation\n"
            "Choice: ")
        K = int(input("Choose numbers of centers: "))
        epochs = int(input("Choose numbers of epochs: "))
        eta = float(input("Choose step (eta). Should be 0.0005 for aprox, and 0.001 for class: "))
        way_of_select_centers = input(
            "In which way choose centers:\n"
            "1. k-means clustering\n"
            "2. random points from data\n"
            "Choice: ")
    else:
        choice = sys.argv[1]
        K = int(sys.argv[2])
        epochs = int(sys.argv[3])
        eta = float(sys.argv[4])
        way_of_select_centers = sys.argv[5]

    if choice == "1":
        # Take data from txt file (x, y, z, t, class)
        file_name = "./Data/classification_train.txt"
        center_eta = 0.0002
    elif choice == "2":
        file_name = "./Data/approximation_train_1.txt"
        center_eta = 0.0003
    else:
        sys.exit()

    data = np.loadtxt(file_name)

    N = np.shape(data)[0]
    data = take_some_lines_from_vector(data.shape[0], data)

    if way_of_select_centers == "1":
        centers_method = "k-means clustering"
        centers, max_distance = choose_centers_with_k_means_clustering(data.T[:-1].T, K)
    elif way_of_select_centers == "2":
        centers_method = "random values from training data"
        center_eta = 0.1
        centers = take_some_lines_from_vector(K, data.T[0])
        max_distance = 0.0
        for centerA in centers:
            for centerB in centers:
                tmp = euclidean_distance(centerA, centerB)
                if max_distance <= tmp:
                    max_distance = np.copy(tmp)
    else:
        sys.exit()

    sigma = np.zeros(K) + (max_distance / (2 * K) ** 0.5)

    data_matrix = make_matrix(data.T[:-1].T, centers, sigma)

    os.system('cls')
    print("Initial data:\n"
          f"number of centers: {K}\n"
          f"epochs: {epochs}\n"
          f"eta: {eta}\n"
          f"choose centers using: {centers_method}\n")

    # In this part we make weights vector, which len is K
    # using Moore–Penrose inverse (np.linalg.pinv() make
    # Moore–Penrose inverse matrix)
    weights = np.dot(np.linalg.pinv(data_matrix), data.T[-1])
    bias = np.random.uniform(-1, 1)

    errors = []
    # Learning
    for epoch in range(epochs):
        data_matrix = make_matrix(data.T[:-1].T, centers, sigma)
        y = np.dot(data_matrix, weights) + bias

        results_differences = y - data.T[-1]

        error = np.sum(results_differences * results_differences) / (2 * N)
        errors = errors + [error]

        delta_weights = np.zeros(weights.shape)
        delta_centers = np.zeros(centers.shape)
        delta_sigma = np.zeros(sigma.shape)
        delta_bias = 0.0

        for i, delta_weight in enumerate(delta_weights):
            for j, results_difference in enumerate(results_differences):
                counted_distance = euclidean_distance(data[j][:-1], centers[i])
                product_of_results_and_gauss = results_difference * gauss_function(counted_distance, sigma[i])

                delta_weights[i] += product_of_results_and_gauss
                delta_centers[i] += product_of_results_and_gauss * (data[j][-1] - centers[i])
                delta_sigma[i] += product_of_results_and_gauss * (counted_distance * counted_distance)
                delta_bias += results_difference

            delta_centers[i] *= (2 * weights[i]) / (sigma[i] ** 2)
            delta_sigma[i] *= (2 * weights[i]) / (sigma[i] ** 3)

        bias = bias - eta * delta_bias
        weights = weights - eta * delta_weights
        centers = centers - eta * center_eta * delta_centers
        sigma = sigma - eta * delta_sigma

    plt.plot(errors)
    plt.grid(b=True)
    plt.show()

    if choice == "1":
        print('Answers pattern: "correct | round e-0 | round e-2" ')
        tests = np.loadtxt("./Data/classification_test.txt")
        obliczone_wartosci = make_matrix(tests.T[:-1].T, centers, sigma).dot(weights) + bias
        licznik = 0
        for i, wartosc in enumerate(obliczone_wartosci):
            if int(tests[i][-1]) == int(round(wartosc, 0)):
                licznik += 1
            print(int(tests[i][-1]), "|", int(round(wartosc, 0)), "|", round(wartosc, 2))

        print(f"Tyle dobrze: {licznik}\nTyle źle: {len(obliczone_wartosci) - licznik}")
        print(f"Skuteczność: {round(licznik / len(obliczone_wartosci) * 100, 2)}%")

    elif choice == "2":
        print(f"Centers {centers.T}")
        print(f"Sigma: {sigma}")

        x_draw = np.loadtxt("./Data/approximation_test.txt").T[0]
        y_draw = np.loadtxt("./Data/approximation_test.txt").T[1]

        plt.text(x=-3, y=0, s=(f'Epochs: {epochs / 1_000}k\n' +
                               r'$\eta$: ' + str(eta) + '\n' +
                               f'Neurons quantity: {K}\n'))
        plt.grid(b=True)
        plt.scatter(x_draw, y_draw, c='g', marker='.')
        plt.scatter(x_draw, make_matrix(x_draw, centers, sigma).dot(weights) + bias, c='r', marker='*')
        plt.scatter(centers, np.zeros(centers.shape), c='b', marker='P', s=150)
        plt.savefig(f"Plots/neurons{K}_file{file_name[-5]}_epochs{epochs // 1000}k.png")
        plt.show()
