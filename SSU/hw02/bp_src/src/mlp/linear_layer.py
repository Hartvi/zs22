import numpy as np


class LinearLayer(object):
    def __init__(self, n_inputs, n_units, rng, name):
        """
        Linear (dense, fully-connected) layer.
        :param n_inputs:
        :param n_units:
        :param rng: random number generator used for initialization
        :param name:
        """
        super(LinearLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.rng = rng
        self.name = name
        # my stuff
        self.output = None
        self.input = None
        self.initialize()

    def has_params(self):
        return True

    def forward(self, X):
        """
        Forward message.
        :param X: layer inputs, shape (n_samples, n_inputs)
        :return: layer output, shape (n_samples, n_units)
        """
        # W doesnt depend on number of samples => multiply from the right hand side
        # W: (n_o, n_i), b: (n_o, )
        # print("W: ", self.W)
        # print("forward pass")
        # print("X.shape: ", X.shape)
        # print("self.W @ X: ", X @ self.W)
        ret = X @ self.W + self.b
        # print("self.output.shape: ", self.output.shape)
        return ret

    def delta(self, Y, delta_next):
        """
        Computes delta (dl/d(layer inputs)), based on delta from the following layer. The computations involve backward
        message.
        :param Y: output of this layer (i.e., input of the next), shape (n_samples, n_units)
        :param delta_next: delta vector backpropagated from the following layer, shape (n_samples, n_units)
        :return: delta vector from this layer, shape (n_samples, n_inputs)
        """
        # y = W @ X + b; dy/dx = W.T; W.T: (n_i, n_o)
        # OK dy/dx must be able to multiply y: in this case W.T @ y: (n_i, n_o) @ (n_o, n_s) = (n_i, n_s)
        # print("delta_next.shape", delta_next.shape)
        # print("self.W.T: ", self.W.T)
        return delta_next @ self.W.T

    def grad(self, X, delta_next):
        """
        Gradient averaged over all samples. The computations involve parameter message.
        :param X: layer input, shape (n_samples, n_inputs)
        :param delta_next: delta vector backpropagated from the following layer, shape (n_samples, n_units)
        :return: a list of two arrays [dW, db] corresponding to gradients of loss w.r.t. weights and biases, the shapes
        of dW and db are the same as the shapes of the actual parameters (self.W, self.b)
        """
        dW = X.T @ delta_next #/ X.shape[0]  # fill lines with x.T weighted by delta
        db = np.ones(X.shape[0]) @ delta_next #/ X.shape[0]
        return [dW, db]

    def initialize(self):
        """
        Perform He's initialization (https://arxiv.org/pdf/1502.01852.pdf). This method is tuned for ReLU activation
        function. Biases are initialized to 1 increasing probability that ReLU is not initially turned off.
        """
        scale = np.sqrt(2.0 / self.n_inputs)
        self.W = self.rng.normal(loc=0.0, scale=scale, size=(self.n_inputs, self.n_units))
        self.b = np.ones(self.n_units)

    def update_params(self, dtheta):
        """
        Updates weighs and biases.
        :param dtheta: contains a two element list of weight and bias updates the shapes of which corresponds to self.W
        and self.b
        """
        assert len(dtheta) == 2, len(dtheta)
        dW, db = dtheta
        assert dW.shape == self.W.shape, dW.shape
        assert db.shape == self.b.shape, "db.shape: " + str(db.shape) + " self.b.shape: "+str(self.b.shape)
        self.W += dW
        self.b += db


if __name__ == "__main__":
    rng = np.random.RandomState(1234)
    X = rng.normal(loc=1.0, scale=1, size=(2, 4))
    lin = LinearLayer(4, 8, rng, "lin_layer")
    print(lin.forward(X))
    print(lin.delta(lin.output, X))
