import numpy as np
class ReLULayer(object):
    def __init__(self, name):
        super(ReLULayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        X[X<=0] = 0
        return X

    def delta(self, Y, delta_next):
        delta_tmp = np.ones(Y.shape)
        delta_tmp[Y==0] = 0
        delta_tmp = np.multiply(delta_next,delta_tmp)
        return delta_tmp
