import numpy as np


class LossCrossEntropy(object):
    def __init__(self, name):
        super(LossCrossEntropy, self).__init__()
        self.name = name
        self.input = None
        self.output = None

    def forward(self, X, T):
        """
        Forward message.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs) (i,s), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs) (s,i)
        :return: layer output, shape (n_samples, 1)
        """
        ret = -np.expand_dims(np.sum(T*np.log(X), axis=1), axis=1)
        # print(ret)
        return ret

    def delta(self, X, T):
        """
        Computes delta vector for the output layer.
        :param X: loss inputs (outputs of the previous layer), shape (n_inputs, n_samples), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_inputs, n_samples)
        :return: delta vector from the loss layer, shape (n_inputs, n_samples)
        """
        # d(-t*log(x)) / d(x) = -t/x
        # print(X.shape, T.shape)
        return -T/X


def softmax(X):
    e_x = np.exp(X.T - np.max(X, axis=1)).T
    ret = (e_x.T / e_x.sum(axis=1)).T
    return ret


class LossCrossEntropyForSoftmaxLogits(object):
    def __init__(self, name):
        super(LossCrossEntropyForSoftmaxLogits, self).__init__()
        self.name = name
        ret = None

    def forward(self, X, T):
        """
        X: (s, i)
        T: (s, i)
        """
        # ln(sum_j^K(x_j))
        exp_max = np.exp(X.T - np.max(X, axis=1)).T
        lns = np.log(np.sum(exp_max, axis=1))  # should be (s, )
        X_times_sum = -X.T + lns
        before_vertical_sum = (T.T*X_times_sum).T  # (i, s) - (s, ) i.e. add the sum columnwise to each element
        ret = before_vertical_sum
        # print("SHAPE:", ret.shape)
        ret = np.sum(before_vertical_sum, axis=1)
        # assert ret.shape == T.shape, str(ret.shape[1])+" != "+ str(T.shape)
        # print("ret.shape: ", ret.shape)
        # print()
        return ret

    def delta(self, X, T):
        """
        X: (s, i)
        T: (s, i)
        ret: (s, i)
        """
        # SOFTMAX - T
        delta = softmax(X) - T
        return delta



if __name__ == "__main__":
    rng = np.random.RandomState(1234)
    X = rng.normal(loc=1.0, scale=1, size=(2, 4))
    lin = LossCrossEntropy("cross_entropy")
    print(lin.forward(X))
    print(lin.delta(lin.output, X))
