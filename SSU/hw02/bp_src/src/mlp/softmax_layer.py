import numpy as np


def vvT(v):
    """
    (n, ) => (1, n) x (n, 1) => (n, n)
    """
    return np.expand_dims(v, axis=1) @ np.expand_dims(v, axis=0)

# '''
class SoftmaxLayer(object):
    def __init__(self, name):
        super(SoftmaxLayer, self).__init__()
        self.name = name
        self.output = None
        self.input = None

    def has_params(self):
        return False

    def forward(self, X):
        e_x = np.exp(X.T - np.max(X, axis=1)).T
        ret = (e_x.T / e_x.sum(axis=1)).T
        return ret

    def delta(self, Y, delta_next):
        """
        Y: (s, o)
        delta_next: (s, o)
        ret: (s, o)
        """
        # -Y @ Y.T + np.eye(Y) but column-wise (if the inputs are vectors)
        delta_new = np.zeros(Y.shape)  # number of inputs = number of outputs
        for i in range(Y.shape[0]):
            y_i = Y[i, :]
            d_i = delta_next[i, :]  # row, i.e. one sample's delta
            dy_dx = -vvT(y_i) + np.diag(y_i)  # TODO: check this
            result = d_i @ dy_dx
            delta_new[i, :] = result

        return delta_new
# '''


if __name__ == "__main__":
    a = np.ones((3, 2))
    a[1, :] = 5
    sf = SoftmaxLayer("softmax_test")
    Y = sf.forward(a)
    print(Y)

    # print(SoftmaxLayer.forward(a))
