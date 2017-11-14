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
    
import time
import h5py
import math

start_time = time.time()

datafile = '/home/elaloy/VAE_MCMC/train_data/maulescreek_unc.hdf5'

def prep_image(im):
    # from (dim1, dim2 dim3) to (dim3, dim1 dim2)
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    return im #floatX(im)

def load_dataset():
    with h5py.File(datafile, 'r') as fid:
        X_train_i = np.array(fid['features'], dtype='uint8')
    
    X_train_i=X_train_i[:,:-1,:,:] # do this for compatibility issue with the vaecnn structure
    # flip axes to go frm nsample,nr,nc,nl to nsample,nl,nr,nc
    X_train=np.zeros((X_train_i.shape[0],X_train_i.shape[3],X_train_i.shape[1],X_train_i.shape[2]),dtype='uint8')
    
    for i in range(0, X_train.shape[0]):
        X_train[i,:,:,:]=prep_image(X_train_i[i,:,:,:])    
    
    print(X_train.shape)
    # for the unc set: X_train[:19500], X_train[19500:]
    # for the con set: X_train[:18000], X_train[18000:]
    X_train, X_test = X_train[:19500], X_train[19500:]
  
    return X_train, X_test
    
X, X_test = load_dataset()       
end_time = time.time()
elapsed_time = end_time - start_time
print("Time getting the data = %5.4f seconds." % (elapsed_time))

print('X type and shape:', X.dtype, X.shape)
print('X.min():', X.min())
print('X.max():', X.max())
print('X_test type and shape:', X_test.dtype, X_test.shape)

X_out = X.reshape((X.shape[0], -1))
print('X_out:', X_out.dtype, X_out.shape)

#from lasagne.layers import Layer
from lasagne.random import get_rng

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# VAE Q layer
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

#% VAECNN network      
conv_num_filters = 16
filter_size = 3
pool_size = 2

pad_in = 'valid'
pad_out = 'full'

input_var = T.tensor4('inputs')
target_var = T.matrix('targets')

encode_hid = 1000
z_hid = 50
decode_hid = encode_hid
ii1=45
ii2=36
dense_upper_mid_size=conv_num_filters*(ii1-2)*(ii2-2)*2
relu_shift=10

input_layer = InputLayer(shape=(None, X.shape[1], X.shape[2], X.shape[3]), input_var=input_var)

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
prediction = T.clip(lasagne.layers.get_output(network),1e-7,1.0-1.e-7)

# Theano functions
def kl_error(mu, log_sigma):  
        return 0.5 * T.sum(1 + 2 * log_sigma - T.exp(2 * log_sigma) - T.sqr(mu), axis = 1)

x_mu = get_output(mu_layer)
x_logs = get_output(log_sigma_layer)

kl_loss = kl_error(x_mu, x_logs).mean()

rec_loss_raw = lasagne.objectives.binary_crossentropy(prediction, target_var)
kl_loss_raw = kl_error(x_mu, x_logs)
rec_loss = lasagne.objectives.binary_crossentropy(prediction, target_var).sum(axis = 1).mean()

loss = rec_loss - kl_loss * 40

params = lasagne.layers.get_all_params(network, trainable=True)

updates = lasagne.updates.adam(loss, params,learning_rate=0.0001)

#test_prediction = lasagne.layers.get_output(network, deterministic=True)
#test_loss = lasagne.objectives.squared_error(test_prediction, target_var)


train_fn = theano.function([input_var, target_var], [loss, rec_loss, kl_loss, 
                                                     kl_loss_raw, rec_loss_raw], updates=updates)

#val_fn = theano.function([input_var, target_var], test_loss) 

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# Training
num_epochs = 100
import time
import sys

dumpfile="vaecnn_tmp.pkl"
Restart=False
if Restart:
    restart_filename="vaecnn_tmp.pkl"      
  
print("Starting training...")

for epoch in range(num_epochs):
    sys.stdout.flush()
    bce_err = 0
    kl_err = 0
    loss_err = 0
    train_batches = 0
    if Restart:
        with open(restart_filename,'rb') as f:
            init_param_values=pickle.load(f)
            lasagne.layers.set_all_param_values(network, init_param_values)
       
    start_time = time.time()
    for batch in iterate_minibatches(X, X_out, 100, shuffle=True):
        inputs, targets = batch
        new_loss, new_rec_loss, new_kl, kl_loss_raw, rec_loss_raw = train_fn(inputs, targets)
        
        if ( ( (math.isnan(new_rec_loss))) | ( (math.isnan(new_kl)))):
            print(new_rec_loss)
            print(new_kl)
            break
        
        bce_err += new_rec_loss
        kl_err += new_kl
        loss_err += new_loss
        train_batches += 1
    if epoch <= num_epochs:
        
        print("Epoch {} of {} took {:.3f}s".format(
            epoch, num_epochs, time.time() - start_time))
        print("  rec loss:\t\t{:.6f}".format(bce_err / train_batches))
        print("  kl loss:\t\t{:.6f}".format(- kl_err / train_batches))
        print("  training loss:\t{:.6f}".format(loss_err / train_batches))
        
    if ( (not (math.isnan(bce_err))) & (not (math.isnan(kl_err)))):
        if not(dumpfile is None):
                    pp=lasagne.layers.get_all_param_values(network)
                    with open(dumpfile, 'wb') as fout:
                        pickle.dump(pp, fout, protocol=-1)
    else:
        print(bce_err)
        print(kl_err)
        break
