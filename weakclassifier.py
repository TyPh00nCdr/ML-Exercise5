class WeakClassifierBase:
    axis = -1

    def __init__(self, theta_threshold):
        self.parity = 1
        self.theta_threshold = theta_threshold

    def predict(self, point):
        return -1 * self.parity if point[self.axis] < self.theta_threshold else 1 * self.parity

    def set_parity(self, parity):
        self.parity = parity

    def __repr__(self):
        return "{}[theta_threshold={}, parity={}]".format(self.__class__.__name__, self.theta_threshold, self.parity)


class WeakClassifierX(WeakClassifierBase):
    axis = 0


class WeakClassifierY(WeakClassifierBase):
    axis = 1
