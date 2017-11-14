from __future__ import print_function
try:
    import cPickle as pickle
except:
    import pickle

import os, sys
sys.setrecursionlimit(10000)

import numpy as np

import theano
import theano.tensor as T
import lasagne

from lasagne.layers import get_output, InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer, MergeLayer
from lasagne.nonlinearities import tanh, sigmoid


from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer

from lasagne.random import get_rng

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class Q_Layer(MergeLayer):
    
    
    def __init__(self, incomings, **kwargs):
        super(Q_Layer, self).__init__(incomings, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        
    def get_output_shape_for(self, input_shapes):
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, log_sigma = inputs
        
        out_shape = mu.shape
        
        return self._srng.normal(out_shape) * T.exp(log_sigma) + mu

def buildnet(weight_file,z_hid = 50):    
 
    conv_num_filters = 16
    filter_size = 3
    pool_size = 2  
    pad_in = 'valid'
    pad_out = 'full' 
    input_var = T.tensor4('inputs')
#    target_var = T.matrix('targets')
    encode_hid = 1000
    decode_hid = encode_hid
    ii1=45
    ii2=36
    dense_upper_mid_size=conv_num_filters*(ii1-2)*(ii2-2)*2
    relu_shift=10
    input_layer = InputLayer(shape=(None, 27, 32, 30), input_var=input_var)
    conv1 = Conv2DLayer(input_layer, num_filters= conv_num_filters, filter_size=filter_size, pad=pad_in)
    conv2 = Conv2DLayer(conv1, num_filters= conv_num_filters, filter_size=filter_size, pad=pad_in)
    pool1 = MaxPool2DLayer(conv2, pool_size=pool_size)
    conv3 = Conv2DLayer(pool1, num_filters= 2*conv_num_filters, filter_size=filter_size, pad=pad_in)
    pool2 = MaxPool2DLayer(conv3, pool_size= pool_size)
    reshape1 = ReshapeLayer(pool2, shape=(([0], -1)))
    encode_h_layer = DenseLayer(reshape1, num_units=encode_hid, nonlinearity=None)
    mu_layer = DenseLayer(encode_h_layer, num_units=z_hid, nonlinearity=None)
    log_sigma_layer = DenseLayer(encode_h_layer, num_units=z_hid, 
                                 nonlinearity = lambda a: T.nnet.relu(a+relu_shift)-relu_shift)
    q_layer = Q_Layer([mu_layer, log_sigma_layer])
    decode_h_layer = DenseLayer(q_layer, num_units=decode_hid, nonlinearity=tanh)
    decode_h_layer_second = DenseLayer(decode_h_layer, num_units=dense_upper_mid_size, nonlinearity=None)
    reshape2 = ReshapeLayer(decode_h_layer_second, shape= ([0], 2*conv_num_filters, (ii1-2), (ii2-2)))
    upscale1 = Upscale2DLayer(reshape2, scale_factor=pool_size)
    deconv1 = Conv2DLayer(upscale1, num_filters=conv_num_filters, filter_size=filter_size, pad=pad_out)
    upscale2 = Upscale2DLayer(deconv1, scale_factor=pool_size)
    deconv2 = Conv2DLayer(upscale2, num_filters=conv_num_filters, filter_size=filter_size, pad=pad_out)
    deconv3 = Conv2DLayer(deconv2, num_filters=1, filter_size=filter_size, pad=pad_out, nonlinearity = sigmoid)
    network = ReshapeLayer(deconv3, shape=(([0], -1)))
        
    with open(weight_file,'rb') as f:
        updated_param_values=pickle.load(f)
        lasagne.layers.set_all_param_values(network, updated_param_values)
    
    encoded_mu = lasagne.layers.get_output(mu_layer)
    ae_encode_mu = theano.function([input_var], encoded_mu)
    encoded_log_sigma = lasagne.layers.get_output(log_sigma_layer)
    ae_encode_log_sigma = theano.function([input_var], encoded_log_sigma)
    x=theano.tensor.matrix()
    mu=theano.tensor.matrix()
    log_sigma=theano.tensor.matrix()
    noise_adjust= theano.function([x,mu,log_sigma], x* T.exp(log_sigma) + mu) 
    noise_var = T.matrix()
    gen = get_output(network, {q_layer:noise_var})
    gen_from_noise = theano.function([noise_var], gen)
    
    def gen_model_from_enc(noise_input,n_steps,gen_from_noise,noise_adjust,
                           ae_encode_mu,ae_encode_log_sigma,threshold=False):
        generated_i = gen_from_noise(noise_input)
        generated = generated_i.reshape(-1, 27, 32, 30)
        for ii in range(0,n_steps):
            mu=ae_encode_mu(generated)
            log_sigma=ae_encode_log_sigma(generated)  
            noise_adj=noise_adjust(noise_input,mu,log_sigma)
            generated = gen_from_noise(noise_adj)
            generated = generated.reshape(-1, 27, 32, 30)
        if threshold:    
            generated[generated<0.5]=0
            generated[generated>=0.5]=1
            X_gen = generated
        else:
            X_gen = generated
        return X_gen    
   
    return ae_encode_mu, ae_encode_log_sigma, noise_adjust, gen_from_noise, gen_model_from_enc
    