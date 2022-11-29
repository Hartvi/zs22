import numpy as np

class SoftmaxLayer(object):
    def __init__(self, name):
        super(SoftmaxLayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        # input bude X: (n_samples, n_inputs)
        # output bude asi taky (n_samples, n_inputs)
        #X[X>10] = 10
        #X[X<-10] = -10
        print("X {}".format(X[0,:]))
        X_exp = np.exp(X)
        softm = X_exp/(np.sum(X_exp, axis=1)).reshape(-1,1)
        print("Y {}".format(softm[0,:]))
        return softm

    def delta(self, Y, delta_next):
        sigmas = Y #self.forward(Y) #????????? Y is already output of this layer presumably ????
        the_whole_shebang = np.zeros(Y.shape)

        # cycle through samples
        for i in range(Y.shape[0]):
            sigmas_sample = sigmas[i,:]
            sigmas_sample_matrix = np.tile(sigmas_sample,(Y.shape[1],1))
            J = -np.multiply(sigmas_sample_matrix,sigmas_sample_matrix.T)
            J = np.diag(sigmas_sample) + J
            #print(J.shape)
            #print(delta_next.shape)
            the_whole_shebang[i,:] = delta_next[i,:] @ J
            
        return the_whole_shebang
