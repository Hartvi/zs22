import numpy as np

# '''
class ReLULayer(object):
    def __init__(self, name):
        super(ReLULayer, self).__init__()
        self.name = name
        self.output = None
        self.input = None

    def has_params(self):
        return False

    def forward(self, X):
        self.input = X
        self.output = X  # if X > 0 else np.zeros(X.shape)
        self.output[self.output < 0] = 0
        return self.output

    def delta(self, Y, delta_next):
        """
        Y: (o, s)
        delta_next: (o, s)
        ret: (i=o, s)
        """
        # should be equivalent to [np.eye(Y[:, i]) @ delta_next[:, i] for i in range(delta_next.shape[1])]
        return delta_next * (Y > 0).astype(int)
# '''

