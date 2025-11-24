from numpy import ndarray, dot, where
from numpy.random import uniform


class AdalineSGD:
    """ Adaptive Linear Neuron classifier

    Parameters
    ------------
    eta: float
        Learning rate (between 0.0 and 1.0)

    n_iter: int
        Passes over the training dataset.

    """

    def __init__(self, eta: float = .001, n_iter: int = 100):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.b = uniform(0, .01)
        self.eps = 1e-5

    def initialize_weights(self, length: int, /):

        self.w_ = uniform(-.01, .01, size=length)

    def fit(self, X: ndarray, y: ndarray, /) -> bool:
        """Fit training data

        Parameters
        -----------

        X:{array-like},
            shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.

        y:-array-like,
            shape = [n_examples]
            Target values.
        """

        self.initialize_weights(X.shape[1])
        n_sample = X.shape[0]
        J_last, J_old = None, None
        for i in range(self.n_iter):
            J_old = J_last
            J_last = 0
            for j in range(n_sample):
                net_input = self.net_input(X[j])
                error = y[j] - net_input
                self.w_ += self.eta * error * X[j]
                self.b += self.eta * error
                J_last += (error ** 2) / 2
            if i != 0:
                if abs(J_last - J_old) <= self.eps:
                    return True
        return False


    def net_input(self, xi, /) -> float:
        """Calculate net input
        Parameters
        ----------

        X: ndarray
            shape = [n_feature]

        Return
        ------

        net_input: float
            net_input = w_ * X + b
        """

        net_input = dot(xi, self.w_) + self.b
        return net_input

    def predict(self, X, /) -> int:
        """Return class label after unit step"""
        result = where(self.net_input(X) > 0, 1, -1)
        return result