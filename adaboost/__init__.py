import matplotlib.pyplot as plt
import numpy as np

from adaboost.weakclassifier import WeakClassifierX, WeakClassifierY


def main():
    data_circle = np.loadtxt('../data/dataCircle.txt')
    data_circle[data_circle[:, 2] == 0, 2] = -1.0

    sorted_x = data_circle[data_circle[:, 0].argsort()]
    sorted_y = data_circle[data_circle[:, 1].argsort()]

    global feats, labels, D, m
    feats = data_circle[:, :2]
    labels = data_circle[:, 2]
    m = len(feats)
    D = np.full(m, 1 / m)

    weak_x = [WeakClassifierX(min(x1[0], x2[0]) + np.abs(x1[0] - x2[0]) / 2) for x1, x2 in
              zip(sorted_x[:], sorted_x[1:]) if x1[2] != x2[2]]
    weak_y = [WeakClassifierY(min(y1[1], y2[1]) + np.abs(y1[1] - y2[1]) / 2) for y1, y2 in
              zip(sorted_y[:], sorted_y[1:]) if y1[2] != y2[2]]

    # AdaBoost
    # ada_boost = AdaBoostClassifier()
    # ada_boost.fit(feats, labels)

    fig, ax = plt.subplots()
    # plot_boundaries(ax, ada_boost)

    # Plot data pointsFs
    ax.scatter(*feats.T, c=labels)
    plt.show()


def epsilon_err(h):
    return sum(D[i] * (1 if h(i) != labels[i] else 0) for i in range(m))


def plot_boundaries(ax, clf):
    """
    See: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
         https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html
    """
    plot_step = 0.02
    x_min, x_max = feats[:, 0].min() - 0.5, feats[:, 0].max() + 0.5
    y_min, y_max = feats[:, 1].min() - 0.5, feats[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    return ax


if __name__ == '__main__':
    main()
