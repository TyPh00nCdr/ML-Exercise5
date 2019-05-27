import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier


def main():
    data_circle = np.loadtxt('data/dataCircle.txt')
    feats = data_circle[:, :2]
    labels = data_circle[:, 2]

    # AdaBoost
    ada_boost = AdaBoostClassifier()
    ada_boost.fit(feats, labels)

    # Plot decision boundaries
    plot_step = 0.02
    x_min, x_max = feats[:, 0].min() - 0.5, feats[:, 0].max() + 0.5
    y_min, y_max = feats[:, 1].min() - 0.5, feats[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    fig, ax = plt.subplots()
    Z = ada_boost.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot data points
    ax.scatter(*feats.T, c=labels)
    plt.show()


if __name__ == '__main__':
    main()
