import numpy as np
from PIL import Image
import glob


# Distance function return euclidean distance
def euclidean_distance(A, B):
    """
    Return euclidean distance in 2D for points A(A_x, A_y) and B(B_x, B_y)

    :return: distance between point A(A_x, A_y) and B(B_x, B_y)

    by: Szymon Jacoń
    """
    return np.sum((A - B) ** 2) ** 0.5


def increse(number):
    number = str(number)
    if len(number) == 5:
        return number
    elif len(number) == 4:
        return "0" + number
    elif len(number) == 3:
        return "00" + number
    elif len(number) == 2:
        return "000" + number
    elif len(number) == 1:
        return "0000" + number


def make_gif(path_to_catalog_with_plots, gif_name):
    """
    This funcion will create gif from images

    :param path_to_catalog_with_plots: enter path to catalog with plots
    :param gif_name: enter name of new file (for examle "myGif" then function will create file "myGif.gif")
    :return:

    by: Szymon Jacoń
    """

    # Create the frames
    frames = []
    imgs = sorted(glob.glob(path_to_catalog_with_plots + "/*"))
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save('../gify/' + gif_name + '.gif', format='GIF', save_all=True, duration=1, loop=0,
                   append_images=frames[1:])


def prepare_data(file_to_change, new_file, delimiter, every_n_row):
    """
    This function is making new shorter file from long old file, loading every n row

    :param file_to_change: enter path to file which u want to change
    :param new_file: enter the path to the file you want to be created
    :param delimiter: can be for example " " or ","
    :param every_n_row: every n lines, which u want to copy
    :return:

    by: Szymon Jacoń
    """

    test_data = np.loadtxt(file_to_change, delimiter=delimiter)

    tmp = []

    for index, value in enumerate(test_data):
        if index % every_n_row == 0:
            tmp.append(value)

    tmp = np.asarray(tmp)

    np.savetxt(new_file, tmp)


def make_matrix(vector1, vector2, sigma):
    matrix = np.empty((len(vector1), len(vector2)))

    for indexA, datumA in enumerate(vector1):
        for indexB, datumB in enumerate(vector2):
            matrix[indexA, indexB] = gauss_function(euclidean_distance(datumA, datumB), sigma[indexB])

    return matrix


def gauss_function(r, sigma_f):
    return np.exp(-(r ** 2) / (2 * sigma_f ** 2))
