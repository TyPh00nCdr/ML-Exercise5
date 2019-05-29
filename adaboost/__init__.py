import matplotlib.pyplot as plt
import numpy as np
from adaboost.classifier import WeakClassifierX, WeakClassifierY, StrongClassifier


def main():
    data_circle = np.loadtxt('../data/dataCircle.txt')
    data_circle[data_circle[:, 2] == 0, 2] = -1.0

    sorted_x = data_circle[data_circle[:, 0].argsort()]
    sorted_y = data_circle[data_circle[:, 1].argsort()]

    global feats, labels, D, m, alpha, Z
    feats = data_circle[:, :2]
    labels = data_circle[:, 2]
    m = len(feats)
    D = np.full(m, 1 / m)

    weak_x = [WeakClassifierX(min(x1[0], x2[0]) + np.abs(x1[0] - x2[0]) / 2) for x1, x2 in
              zip(sorted_x[:], sorted_x[1:]) if x1[2] != x2[2]]
    weak_y = [WeakClassifierY(min(y1[1], y2[1]) + np.abs(y1[1] - y2[1]) / 2) for y1, y2 in
              zip(sorted_y[:], sorted_y[1:]) if y1[2] != y2[2]]
    weak_classifiers = weak_x + weak_y
    strong_classifier = StrongClassifier()

    # Gedanken: Kann selber weak learner 2x ausgewählt werden? Falls ja, wird dann nur sein alpha im strong learner ge-
    # updated oder wird eine Kopie des weak learners mit neuem alpha zum strong learner hinzugefügt? Mit letzterem würde
    # auch der Error-Wert zum Schritt t gemerkt und der Error könnte korrekt berechnet werden (nicht nur vom aktuellen
    # Error ausgehend, wie aktuell!!
    for t in range(2000):
        for clf in weak_classifiers:
            if epsilon_err(clf) >= .5:
                clf.flip_parity()
                clf.error = epsilon_err(clf)
            else:
                clf.error = epsilon_err(clf)
        next_classifier = min(weak_classifiers, key=lambda h: h.error)
        next_classifier.alpha = .5 * np.log((1 - next_classifier.error) / next_classifier.error)
        # weak_classifiers.remove(next_classifier)
        strong_classifier.add_weak_clf(next_classifier)
        # Z = sum(D[i] * np.exp(-1 * next_classifier.alpha * feat[2] * next_classifier.predict(feat)) for i, feat in enumerate(data_circle))
        Z = np.sqrt(1 - 4 * (.5 - next_classifier.error) ** 2)
        D = [(1 / Z) * D[i] * np.exp(-1 * next_classifier.alpha * feat[2] * next_classifier.predict(feat)) for
             i, feat in enumerate(data_circle)]
        print(np.exp(-2 * sum((-2 * (0.5 - clf.error) ** 2 for clf in
                               weak_classifiers))))  # wrong: error always uptodate and not historical

    # print([strong_classifier.predict(i) for i in feats])
    # print(labels)

    # AdaBoost
    # ada_boost = AdaBoostClassifier()
    # ada_boost.fit(feats, labels)

    fig, ax = plt.subplots()
    # plot_boundaries(ax, ada_boost)
    for clf in strong_classifier.weak_classifiers:
        if isinstance(clf, WeakClassifierX):
            ax.plot(np.linspace(-10, 10, num=2), np.full(2, clf.theta_threshold), 'k--')
        elif isinstance(clf, WeakClassifierY):
            ax.plot(np.full(2, clf.theta_threshold), np.linspace(-10, 10, num=2), 'k--')

    # Plot data pointsFs
    ax.scatter(*feats.T, c=labels)
    plt.show()


def epsilon_err(h):
    return sum([D[i] * (h.predict(feat) != labels[i]) for i, feat in enumerate(feats)])


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
