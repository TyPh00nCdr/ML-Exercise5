import numpy as np


class WeakClassifierBase:
    axis = -1

    def __init__(self, theta_threshold):
        self.parity = 1
        self.theta_threshold = theta_threshold
        self.alpha = 0
        self.error = 0

    def predict(self, point):
        return -1 * self.parity if point[self.axis] < self.theta_threshold else 1 * self.parity

    def flip_parity(self):
        self.parity = self.parity * -1

    def __repr__(self):
        return "{}[theta_threshold={}, parity={}]".format(self.__class__.__name__, self.theta_threshold, self.parity)


class WeakClassifierX(WeakClassifierBase):
    axis = 0


class WeakClassifierY(WeakClassifierBase):
    axis = 1


class StrongClassifier:
    weak_classifiers = []

    def predict(self, point):
        ret = np.sign(sum(clf.alpha * clf.predict(point) for clf in self.weak_classifiers))
        assert ret != 0, "Error occurred: Label was 0 instead or either -1 or 1"
        return ret

    def add_weak_clf(self, clf):
        if clf not in self.weak_classifiers:
            self.weak_classifiers.append(clf)
