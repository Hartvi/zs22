import numpy as np
from sklearn.neural_network import _multilayer_perceptron


# '''
class LossCrossEntropy(object):
    def __init__(self, name):
        super(LossCrossEntropy, self).__init__()
        self.name = name
        self.output = None
        self.input = None

    def forward(self, X, T):
        """
        Forward message.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs) (i,s), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs) (s,i)
        :return: layer output, shape (n_samples, 1)
        """
        self.input = X
        self.output = -np.expand_dims(np.sum(T*np.log(X), axis=1), axis=1)
        # print(self.output)
        return self.output

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
# '''


class LossCrossEntropyForSoftmaxLogits(object):
    def __init__(self, name):
        super(LossCrossEntropyForSoftmaxLogits, self).__init__()
        self.name = name
        self.output = None

    def forward(self, X, T):
        """
        X: (i, s)
        T: (i, s)
        """
        # ln(sum_j^K(x_j))
        exp_max = np.exp(X.T - np.max(X, axis=1)).T
        lns = np.log(np.sum(exp_max, axis=1))  # should be (s, )
        # print("lns.shape: ", lns.shape)
        # print("T.shape: ", T.shape)  # checked and they are the same
        # print("X.shape: ", X.shape)  # checked and they are the same
        X_times_sum = -X.T + lns
        # print("X_times_sum.shape: ", X_times_sum.shape)
        before_vertical_sum = (T.T*X_times_sum).T  # (i, s) - (s, ) i.e. add the sum columnwise to each element
        # print("before_vertical_sum: ", before_vertical_sum.shape)
        self.output = before_vertical_sum
        # self.output = np.sum(before_vertical_sum, axis=1)
        assert self.output.shape == T.shape, str(self.output.shape[1])+" != "+ str(T.shape)
        # print("self.output.shape: ", self.output.shape)
        # print()
        return self.output

    def delta(self, X, T):
        """
        X: (i, s)
        T: (i, s)
        ret: (i, s)
        """
        # print("\nX.shape: ", X.shape)
        # print("T.shape: ", T.shape)
        exp_X = np.exp(X.T - np.max(X, axis=1)).T
        summies = np.sum(exp_X, axis=1)  # should be (s, )
        inv_summies = 1.0 / summies
        # print("summies.shape: ", summies.shape)
        # print("exp_X.shape: ", exp_X.shape)
        # print("T.shape: ", T.shape)
        # print("((summies - exp_X.T)*inv_summies).T.shape: ", ((summies - exp_X.T)*inv_summies).T.shape)
        delta = T*((summies - exp_X.T)*inv_summies).T
        return delta



if __name__ == "__main__":
    rng = np.random.RandomState(1234)
    X = rng.normal(loc=1.0, scale=1, size=(2, 4))
    lin = LossCrossEntropy("cross_entropy")
    print(lin.forward(X))
    print(lin.delta(lin.output, X))
