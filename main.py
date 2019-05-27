import numpy as np
import sklearn
import matplotlib.pyplot as plt


def main():
    data_circle = np.loadtxt('data/dataCircle.txt')
    fig, ax = plt.subplots()
    ax.scatter(*data_circle[:, :2].T, c=data_circle[:, 2])
    plt.show()


if __name__ == '__main__':
    main()
