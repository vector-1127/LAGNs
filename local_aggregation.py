from keras.layers.core import *
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, regularizers, constraints


class MyPooling(Layer):
    '''
    Input shape: (batch_size, dim, channel)
    Output shape: (batch_size, dim, 1)
    '''
    def __init__(self, pool_way = 'my_mean', **kwargs):
        self.pool_way = getattr(self, pool_way)
        super(MyPooling, self).__init__(**kwargs)
                
    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.dim = input_shape[1]
        self.channel = input_shape[2]
        self.filters = input_shape[3]

    def call(self, x, mask = None):
        return self.pool_way(x)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)
    
    def my_mean(self, x):
        return K.mean(x, axis = 3)
        
    def my_max(self, x):
        return K.max(x, axis = 3)


class LocalNet(Layer):
    '''
    Input shape: (batch_size, dim, nb_channel)
    Output shape: (batch_size, dim, nb_filter)
    '''
    def __init__(self, nb_polynomial_order, neighbor_index, neighbor_weight, neighbor_field, my_init = 0.001,
                 mult = True, learnable = True, fitting_function = 'chebyshev', **kwargs):
        if K.backend() != 'tensorflow':
            raise Exception("GraphConv Requires Tensorflow Backend.")
        self.nb_polynomial_order = nb_polynomial_order
        self.neighbor_index = neighbor_index
        self.neighbor_weight = neighbor_weight
        self.neighbor_field = neighbor_field
        self.my_init = my_init
        self.mult = mult
        self.learnable = learnable
        self.fitting_function = getattr(self, fitting_function)
        super(LocalNet, self).__init__(**kwargs)
        
                
    def build(self, input_shape):
        self.dim = input_shape[1]
        self.nb_channel = input_shape[2]
        if self.mult:
            self.W_poly_shape = (self.nb_channel, self.nb_polynomial_order)
        else:
            self.W_poly_shape = (self.nb_polynomial_order,)
        
        if self.learnable:
            self.W_poly = K.random_uniform_variable(self.W_poly_shape,-1*self.my_init, 1*self.my_init,
                                                    name='{}_W_poly'.format(self.name))
            self.trainable_weights = [self.W_poly]
        else:
            self.W_poly = K.random_uniform_variable(self.W_poly_shape,1,1,
                                                    name='{}_W_poly'.format(self.name))
            self.trainable_weights = []

        self.built = True
        
        
    def call(self, x, mask = None):
        # x shape: (batch_size, dim, channel) => (batch_size, dim, channel, nb_polynomial_order)
        y = self.fitting_function(x)
        # (batch_size, dim, channel, nb_polynomial_order)*{(nb_channel, nb_polynomial_order) or (nb_polynomial_order, )}
        # (batch_size, dim, channel, nb_polynomial_order)
        output = tf.multiply(y, self.W_poly)
        return self.Agg(output)#tf.transpose(output, (0,1,3,2))
        
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim, self.nb_polynomial_order, self.nb_channel)
        
    def Agg(self, x):
        # (batch_size, dim, channel, nb_polynomial_order) => (dim, channel, nb_polynomial_order, batch_size)
        x = tf.transpose(x, perm = (1,2,3,0))
        # (dim, nb_neighbor, channel, nb_polynomial_order, batch_size)
        # (dim, channel, nb_polynomial_order, batch_size)
        y = K.sum(tf.gather(x, self.neighbor_index), axis = 1)
        
        # (batch_size, channel, nb_polynomial_order, dim)
        z = tf.transpose(y, perm = (3,1,2,0))
        z = z*self.neighbor_field
        # (batch_size, dim, nb_polynomial_order, channel)
        z = tf.transpose(z, perm = (0,3,2,1))
        return z
    
        
    def chebyshev(self, x):
        # Chebyshev polynomial: [-1,1]
        x = tf.expand_dims(x, dim = 3)
        if self.nb_polynomial_order > 1:
            x0 = tf.ones_like(x, dtype = 'float32')
            x1 = 2 * x # x or 2 * x
            y = tf.concat(axis = 3, values = [x0,x1])
        for _ in range(2, self.nb_polynomial_order):
            x2 = 2 * x * x1 - x0
            y = tf.concat(axis = 3, values = [y,x2])
            x0, x1 = x1, x2
        return y
    
    def legendre(self,x):
        # Legendre Polynomials : [-1,1]
        x = tf.expand_dims(x, dim = 3)
        if self.nb_polynomial_order > 1:
            x0 = tf.ones_like(x, dtype = 'float32')
            x1 = x
            y = tf.concat(axis = 3, values = [x0,x1])
        for n in range(2, self.nb_polynomial_order):
            x2 = 1.0*(2*n-1)/(n) * x * x1 - 1.0*(n-1)/(n)*x0
            y = tf.concat(axis = 3, values = [y,x2])
            x0, x1 = x1, x2
        return y
        
    def laguerre(self,x):
        # Laguerre polynomial: [0,+infinity]
        x = tf.expand_dims(x, dim = 3)
        if self.nb_polynomial_order > 1:
            x0 = tf.ones_like(x, dtype = 'float32')
            x1 = x0 - x
            y = tf.concat(axis = 3, values = [x0,x1])
        for n in range(2, self.nb_polynomial_order):
            x2 = (2 * n -1) * x1 - x * x1 - (n-1) * (n-1) * x0
            y = tf.concat(axis = 3, values = [y,x2])
            x0, x1 = x1, x2
        return y
        
    def hermite(self, x):
        # Hermite polynomial: [-infinity,+infinity]
        x = tf.expand_dims(x, dim = 3)
        if self.nb_polynomial_order > 1:
            x0 = tf.ones_like(x, dtype = 'float32')
            x1 = 2 * x
            y = tf.concat(axis = 3, values = [x0,x1])
        for n in range(2, self.nb_polynomial_order):
            x2 = 2 * x * x1 - 2 * (n-1) * x0
            y = tf.concat(axis = 3, values = [y,x2])
            x0, x1 = x1, x2
        return y
        
    def optimum(self, x):
        # Optimum polynomial: {1, x, x^{2}, x^{3}, ...,}
        x = tf.expand_dims(x, dim = 3)
        if self.nb_polynomial_order > 1:
            x0 = tf.ones_like(x, dtype = 'float32')
            x1 = x
            y = tf.concat(axis = 3, values = [x0,x1])
        for n in range(2, self.nb_polynomial_order):
            x2 = x * x1
            y = tf.concat(axis = 3, values = [y,x2])
            x0, x1 = x1, x2
        return y
















