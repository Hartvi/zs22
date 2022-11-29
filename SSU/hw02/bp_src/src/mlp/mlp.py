import numpy as np
import matplotlib.pyplot as plt

# '''
class MLP(object):
    def __init__(self, n_inputs, layers, loss, output_layers=[]):
        """
        MLP constructor.
        :param n_inputs:
        :param layers: list of layers
        :param loss: loss function layer
        :param output_layers: list of layers appended to "layers" in evaluation phase, parameters of these are not used
        in training phase
        """
        self.n_inputs = n_inputs
        self.layers = layers
        self.output_layers = output_layers
        self.loss = loss
        self.first_param_layer = layers[-1]
        for l in layers:
            if l.has_params():
                self.first_param_layer = l
                break

    def propagate(self, X, output_layers=True, last_layer=None):
        """
        Feedforwad network propagation
        :param X: input data, shape (n_samples, n_inputs)
        :param output_layers: controls whether the self.output_layers are appended to the self.layers in evaluatin
        :param last_layer: if not None, the propagation will stop at layer with this name
        :return: propagated inputs, shape (n_samples, n_units_of_the_last_layer)
        """
        layers = self.layers + (self.output_layers if output_layers else [])
        if last_layer is not None:
            assert isinstance(last_layer, str)
            layer_names = [layer.name for layer in layers]
            layers = layers[0: layer_names.index(last_layer) + 1]
        for layer in layers:
            X = layer.forward(X)
        return X

    def propagate_for_grads(self, X, output_layers=True, last_layer=None):
        """
        Feedforwad network propagation
        :param X: input data, shape (n_samples, n_inputs)
        :param output_layers: controls whether the self.output_layers are appended to the self.layers in evaluatin
        :param last_layer: if not None, the propagation will stop at layer with this name
        :return: propagated inputs, shape (n_samples, n_units_of_the_last_layer)
        """
        layers = self.layers + (self.output_layers if output_layers else [])
        if last_layer is not None:
            assert isinstance(last_layer, str)
            layer_names = [layer.name for layer in layers]
            layers = layers[0: layer_names.index(last_layer) + 1]
        for layer in layers:
            layer.input = X
            X = layer.forward(X)
            layer.output = X
        return X

    def evaluate(self, X, T):
        """
        Computes loss.
        :param X: input data, shape (n_samples, n_inputs)
        :param T: target labels, shape (n_samples, n_outputs)
        :return:
        """
        # print("inside loss forward")
        # print("T.shape: ", T.shape)
        # print("self.loss.forward(self.propagate(X, output_layers=False), T): ", self.loss.forward(self.propagate(X, output_layers=False), T).shape)
        ret = self.loss.forward(self.propagate(X, output_layers=False), T)
        return ret

    def gradient(self, X, T):
        """
        Computes gradient of loss w.r.t. all network parameters.
        :param X: input data, shape (n_inputs, n_samples)
        :param T: target labels, shape (n_outputs, n_samples)
        :return: a dict of records in which key is the layer.name and value the output of grad function
        """
        # """
        self.propagate_for_grads(X, output_layers=False)
        grads = dict()  # return this
        # first generate loss, then backprop from there
        layers = list(self.layers)
        num_of_layers = len(layers)
        deltas = [None for _ in range(num_of_layers)]
        deltas.append(self.loss.delta(self.layers[-1].output, T))
        for i in range(num_of_layers-1, -1, -1):
            layer = layers[i]
            delta_next = deltas[i+1]  # from last to first
            new_delta = layer.delta(layer.output, delta_next)
            deltas[i] = new_delta
            if i > 0:
                prev_layers_output_ie_this_layers_input = layers[i-1].output
            else:
                prev_layers_output_ie_this_layers_input = X
            if layer.has_params():
                grads[layer.name] = layer.grad(prev_layers_output_ie_this_layers_input, delta_next)
        return grads
        # """
# '''


