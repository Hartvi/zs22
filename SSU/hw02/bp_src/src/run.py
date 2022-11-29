import pickle
import numpy as np
import matplotlib.pyplot as plt

from mlp.datasets import load_XOR, load_spirals, load_MNIST, plot_2D_classification
from mlp.linear_layer import LinearLayer
from mlp.relu_layer import ReLULayer
from mlp.softmax_layer import SoftmaxLayer
from mlp.losses import LossCrossEntropy, LossCrossEntropyForSoftmaxLogits
from mlp.train import train

from mlp.mlp import MLP

def plot_convergence(run_info):
    plt.plot(run_info['acc_train'], label='train')
    plt.plot(run_info['acc_test'], label='test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()


def plot_test_accuracy_comparison(run_info_dict):
    keys = sorted(run_info_dict.keys())
    for key in keys:
        plt.plot(run_info_dict[key]['acc_test'], label=key)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()


def experiment_XOR():
    X, T = load_XOR()
    # X, T = X.T, T.T
    rng = np.random.RandomState(1234)
    net = MLP(n_inputs=2,
              layers=[
                  LinearLayer(n_inputs=2, n_units=4, rng=rng, name='Linear_1'),
                  ReLULayer(name='ReLU_1'),
                  LinearLayer(n_inputs=4, n_units=2,
                              rng=rng, name='Linear_OUT'),
                  SoftmaxLayer(name='Softmax_OUT')
              ],
              loss=LossCrossEntropy(name='CE'),
              )

    run_info = train(net, X, T, batch_size=4, alpha=0.1,
                     n_epochs=100, verbose=False)
    plot_convergence(run_info)
    plt.show()
    print("X:", X)
    print(net.propagate(X))
    print("propagated")
    plot_2D_classification(X, T, net)
    plt.show()


def experiment_spirals():
    X_train, T_train, X_test, T_test = load_spirals()
    experiments = (
        ('alpha = 0.01', 0.01),
        ('alpha = 1', 1.0),
        ('alpha = 5', 5.0),
    )
    run_info_dict = {}
    for name, alpha in experiments:
        rng = np.random.RandomState(1234)
        net = MLP(n_inputs=2,
                  layers=[
                      LinearLayer(n_inputs=2, n_units=10,
                                  rng=rng, name='Linear_1'),
                      ReLULayer(name='ReLU_1'),
                      LinearLayer(n_inputs=10, n_units=3,
                                  rng=rng, name='Linear_OUT'),
                      SoftmaxLayer(name='Softmax_OUT')
                  ],
                  loss=LossCrossEntropy(name='CE'),
                  )

        run_info = train(net, X_train, T_train, batch_size=len(X_train), alpha=alpha, X_test=X_test, T_test=T_test,
                         n_epochs=1000, verbose=True)
        run_info_dict[name] = run_info
        # plot_spirals(X_train, T_train, net)
        # plt.show()
        # plot_convergence(run_info)
        # plt.show()
    plot_test_accuracy_comparison(run_info_dict)
    plt.show()
    # plt.savefig('spiral.pdf') # you can instead save figure to file


def experiment_MNIST_unstable():
    X_train, T_train, X_test, T_test = load_MNIST()
    np.seterr(all='raise', under='warn', over='warn')
    rng = np.random.RandomState(1234)
    net = MLP(n_inputs=28 * 28,
              layers=[
                  LinearLayer(n_inputs=28 * 28, n_units=64,
                              rng=rng, name='Linear_1'),
                  ReLULayer(name='ReLU_1'),
                  LinearLayer(n_inputs=64, n_units=64,
                              rng=rng, name='Linear_2'),
                  ReLULayer(name='ReLU_2'),
                  LinearLayer(n_inputs=64, n_units=64,
                              rng=rng, name='Linear_3'),
                  ReLULayer(name='ReLU_3'),
                  LinearLayer(n_inputs=64, n_units=64,
                              rng=rng, name='Linear_4'),
                  ReLULayer(name='ReLU_4'),
                  LinearLayer(n_inputs=64, n_units=64,
                              rng=rng, name='Linear_5'),
                  ReLULayer(name='ReLU_5'),
                  LinearLayer(n_inputs=64, n_units=10,
                              rng=rng, name='Linear_OUT'),
                  SoftmaxLayer(name='Softmax_OUT')
              ],
              loss=LossCrossEntropy(name='CE'),
              )

    run_info = train(net, X_train, T_train, batch_size=3000, alpha=1e-1,
                     X_test=X_test, T_test=T_test, n_epochs=10, verbose=True)


def experiment_MNIST():
    X_train, T_train, X_test, T_test = load_MNIST()
    np.seterr(all='raise', under='warn', over='warn')
    rng = np.random.RandomState(1234)
    net = MLP(n_inputs=28 * 28,
              layers=[
                  LinearLayer(n_inputs=28 * 28, n_units=64,
                              rng=rng, name='Linear_1'),
                  ReLULayer(name='ReLU_1'),
                  LinearLayer(n_inputs=64, n_units=64,
                              rng=rng, name='Linear_2'),
                  ReLULayer(name='ReLU_2'),
                  LinearLayer(n_inputs=64, n_units=64,
                              rng=rng, name='Linear_3'),
                  ReLULayer(name='ReLU_3'),
                  LinearLayer(n_inputs=64, n_units=64,
                              rng=rng, name='Linear_4'),
                  ReLULayer(name='ReLU_4'),
                  LinearLayer(n_inputs=64, n_units=64,
                              rng=rng, name='Linear_5'),
                  ReLULayer(name='ReLU_5'),
                  LinearLayer(n_inputs=64, n_units=10,
                              rng=rng, name='Linear_OUT'),
              ],
              loss=LossCrossEntropyForSoftmaxLogits(name='CE'),
              output_layers=[SoftmaxLayer(name='Softmax_OUT')]
              )

    run_info = train(net, X_train, T_train, batch_size=3000, alpha=1e-1, X_test=X_test, T_test=T_test, n_epochs=100,
                     verbose=True)
    # plot_convergence(run_info)
    # plt.show()

    with open('MNIST_run_info.p', 'wb') as f:
        pickle.dump(run_info, f)


if __name__ == '__main__':
    # experiment_XOR()

    experiment_spirals()

    # experiment_MNIST_unstable()

    # experiment_MNIST()
