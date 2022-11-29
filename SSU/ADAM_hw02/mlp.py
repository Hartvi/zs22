import numpy as np
""" import matplotlib
matplotlib.use('TkAgg') """

import matplotlib.pyplot as plt

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

    def propagate_with_memory(self, X, output_layers=True):
        """
        Feedforwad network propagation
        :param X: input data, shape (n_samples, n_inputs)
        :param output_layers: controls whether the self.output_layers are appended to the self.layers in evaluatin
        :param last_layer: if not None, the propagation will stop at layer with this name
        :return: propagated inputs, shape (n_samples, n_units_of_the_last_layer)
        """
        self.layer_inputs = dict()
        self.layer_outputs = dict()

        layers = self.layers + (self.output_layers if output_layers else [])

        for layer in layers:
            self.layer_inputs[layer.name] = X
            X = layer.forward(X)
            self.layer_outputs[layer.name] = X


    def evaluate(self, X, T):
        """
        Computes loss.
        :param X: input data, shape (n_samples, n_inputs)
        :param T: target labels, shape (n_samples, n_outputs)
        :return:
        """
        return self.loss.forward(self.propagate(X, output_layers=False), T)

    def gradient(self, X, T):
        """
        Computes gradient of loss w.r.t. all network parameters.
        :param X: input data, shape (n_samples, n_inputs)
        :param T: target labels, shape (n_samples, n_outputs)
        :return: a dict of records in which key is the layer.name and value the output of grad function
        """
        self.propagate_with_memory(X, output_layers=False)

        #print([self.layer_inputs[k][0,:] for k in self.layer_inputs.keys()])
        #print([self.layer_outputs[k][0,:] for k in self.layer_outputs.keys()])

        ret_dict = dict()

        delta_next = self.loss.delta(self.layer_outputs[self.layers[-1].name], T)
        
        for layer in self.layers[::-1]:

            if layer.has_params():
                ret_dict[layer.name] = layer.grad(self.layer_inputs[layer.name], delta_next)

            delta_next = layer.delta(self.layer_outputs[layer.name], delta_next)

        #print(ret_dict)
        return ret_dict
