import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier


def main():
    data_circle = np.loadtxt('data/dataCircle.txt')
    global feats, labels
    feats = data_circle[:, :2]
    labels = [-1 if i == 0 else i for i in data_circle[:, 2]]

    # AdaBoost
    ada_boost = AdaBoostClassifier()
    ada_boost.fit(feats, labels)

    fig, ax = plt.subplots()
    plot_boundaries(ax, ada_boost)

    # Plot data pointsFs
    ax.scatter(*feats.T, c=labels)
    plt.show()


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
