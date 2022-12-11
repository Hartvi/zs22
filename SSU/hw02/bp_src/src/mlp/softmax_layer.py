import pickle
import numpy as np


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
        dY_dX = np.apply_along_axis(func1d=lambda x: np.diag(x) - np.outer(x, x), axis=1, arr=Y)
        
        # could do concat over the axis and then multiply but it's just a for loop in disguise
        # delta_new = delta_next @ dY_dX  # basically this but sliced up
        
        # delta_new = np.dot(delta_next, dY_dX)
        # delta_new = np.diagonal(delta_new).T  # 2.5% faster
        delta_new = np.zeros(Y.shape)  # number of inputs = number of outputs
        for i in range(Y.shape[0]):
            d_i = delta_next[i]  # row, i.e. one sample's delta
            # dy_dx = -vvT(y_i) + np.diag(y_i)
            delta_new[i] = d_i @ dY_dX[i]
        return delta_new


if __name__ == "__main__":
    a = np.ones((3, 2))
    a[1, :] = 5
    sf = SoftmaxLayer("softmax_test")
    Y = sf.forward(a)
    print(Y)
    print(np.outer([1,2,3], [1,2,3]))

    # print(SoftmaxLayer.forward(a))

    with open('MNIST_run_info.p', 'rb') as f:
        print(pickle.load(f).keys())
    
    import matplotlib.pyplot as plt
    epoch_weights = np.load('epoch_weights.npy')
    epochs = epoch_weights.shape[0]
    x_axis = np.arange(epochs)
    Ws = epoch_weights[:,:,0]
    Ws /= Ws[0, :]
    bs = epoch_weights[:,:,1]
    bs /= bs[0, :]
    layers = Ws.shape[1]
    # print(bs)
    # ax1 = plt.subplot(1, 2, 1)
    # ax2 = plt.subplot(1, 2, 2)
    for i in range(layers):
        # ax1.plot(x_axis, Ws[:, i], color=(float(i)/(layers-1), 0.5, 0.0), label=f'Layer{i+1} W')
        plt.plot(x_axis, Ws[:, i], label=f'Layer{i+1} Weight')
        # plt.plot(x_axis, bs[:, i], label=f'Layer{i+1} Bias')
        # ax2.plot(x_axis, bs[:, i], color=(0.2,float(i)/layers, 1-float(i)/(layers-1)), label=f'Layer{i+1} b')
    plt.title("Mean Weight Amplitudes of Layer Weights by Epoch")
    plt.ylabel("Mean weight amplitude")
    plt.legend()
    plt.xlabel("Epoch")
    plt.show()
    print()
