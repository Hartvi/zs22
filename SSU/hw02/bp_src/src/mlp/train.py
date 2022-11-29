import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

def accuracy(Y, T):
    p = np.argmax(Y, axis=1)
    t = np.argmax(T, axis=1)
    return np.mean(p == t)


def train(net, X_train, T_train, batch_size=1, n_epochs=2, alpha=0.1, X_test=None, T_test=None, verbose=False):
    """
    Trains a network using vanilla gradient descent.
    :param net:
    :param X_train:
    :param T_train:
    :param batch_size:
    :param n_epochs:
    :param alpha: learning rate
    :param X_test:
    :param T_test:
    :param verbose: prints evaluation for each epoch if True
    :return:
    """
    n_samples = X_train.shape[0]
    assert T_train.shape[0] == n_samples
    assert batch_size <= n_samples
    run_info = defaultdict(list)

    def process_info(epoch):
        loss_test, acc_test = np.nan, np.nan
        Y = net.propagate(X_train, output_layers=False)
        loss_train = net.loss.forward(Y, T_train)
        acc_train = accuracy(Y, T_train)
        run_info['loss_train'].append(loss_train)
        run_info['acc_train'].append(acc_train)
        if X_test is not None:
            Y = net.propagate(X_test, output_layers=False)
            loss_test = net.loss.forward(Y, T_test)
            acc_test = accuracy(Y, T_test)
            run_info['loss_test'].append(loss_test)
            run_info['acc_test'].append(acc_test)
        if verbose:
            print('epoch: {}, loss: {}/{} accuracy: {}/{}'.format(epoch, np.mean(loss_train), np.nanmean(loss_test),
                                                                  np.nanmean(acc_train), np.nanmean(acc_test)))

    process_info('initial')
    for epoch in range(1, n_epochs + 1):
        offset = 0
        while offset < n_samples:
            last = min(offset + batch_size, n_samples)
            if verbose:
                print('.', end='')
            grads = net.gradient(np.asarray(X_train[offset:last]), np.asarray(T_train[offset:last]))
            for layer in net.layers:
                if layer.has_params():
                    gs = grads[layer.name]
                    dtheta = [-alpha * g for g in gs]
                    layer.update_params(dtheta)

            offset += batch_size
        if verbose:
            print()
        process_info(epoch)
    return run_info
