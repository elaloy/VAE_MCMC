from __future__ import print_function
try:
    import cPickle as pickle
except:
    import pickle

import os, sys
sys.setrecursionlimit(10000)

import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
import lasagne

from lasagne.layers import get_output, InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer, MergeLayer
from lasagne.nonlinearities import tanh, sigmoid

from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
   
import time
import h5py
from lasagne.random import get_rng

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#%% Select case study and load training data
# 1: Channelized aquifer, no direct conditioning data
# 2: Channelized aquifer, including 9 direct conditioning data
# 3: 3D Maules Creek aquifer, no direct conditioning data
# 4: 3D Maules Creek aquifer, including 56 direct conditioning data
case_study=1

start_time = time.time()

# Load a subset of the training data to perform checks
data_dir='home/elaloy/VAE_MCMC/train_data'
#data_dir='D:/OwnResearch/DeepLearning/Python/emul/emul'
if case_study==1:
    datafile=data_dir+'/channel_unc_dataset.hdf5'
    def load_dataset():
        with h5py.File(datafile, 'r') as fid:
            X_train = np.array(fid['features'], dtype='uint8')
        print(X_train.shape)
        X_train, X_test = X_train[0:5000], X_train[80000:83000]
        return X_train, X_test
    X, X_test = load_dataset()          
if case_study==2:
    datafile=data_dir+'/channel_con_dataset.hdf5'
    def load_dataset():
        with h5py.File(datafile, 'r') as fid:
            X_train = np.array(fid['features'], dtype='uint8')
        print(X_train.shape)
        X_train, X_test = X_train[:39500], X_train[39500:]
        return X_train, X_test
    X, X_test = load_dataset()          
if case_study==3:
    datafile=data_dir+'/maulescreek_unc.hdf5'
    def prep_image(im):
        # from (dim1, dim2 dim3) to (dim3, dim1 dim2)
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
        X_train, X_test = X_train[:19500], X_train[19500:]
        return X_train, X_test   
    X, X_test = load_dataset()       
if case_study==4:
    datafile=data_dir+'/maulescreek_con.hdf5'
    def prep_image(im):
        # from (dim1, dim2 dim3) to (dim3, dim1 dim2)
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
        X_train, X_test = X_train[:18000], X_train[18000:]
        return X_train, X_test   
    X, X_test = load_dataset()       
 
end_time = time.time()
elapsed_time = end_time - start_time
print("Time getting the data = %5.4f seconds." % (elapsed_time))

print('X type and shape:', X.dtype, X.shape)
print('X.min():', X.min())
print('X.max():', X.max())

X_out = X.reshape((X.shape[0], -1))
print('X_out:', X_out.dtype, X_out.shape)

#% Build generator - step 1
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

#% Build generator - step 2
if case_study == 1 or case_study == 2:    
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
    dense_upper_mid_size=conv_num_filters*23*23*2
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
    reshape2 = ReshapeLayer(decode_h_layer_second, shape= ([0], 2*conv_num_filters, 23, 23))
    
    upscale1 = Upscale2DLayer(reshape2, scale_factor=pool_size)
    deconv1 = Conv2DLayer(upscale1, num_filters=conv_num_filters, filter_size=filter_size, pad=pad_out)
    upscale2 = Upscale2DLayer(deconv1, scale_factor=pool_size)
    deconv2 = Conv2DLayer(upscale2, num_filters=conv_num_filters, filter_size=filter_size, pad=pad_out)
    deconv3 = Conv2DLayer(deconv2, num_filters=1, filter_size=filter_size, pad=pad_out, nonlinearity = sigmoid)
    
    network = ReshapeLayer(deconv3, shape=(([0], -1)))
    
    prediction = T.clip(lasagne.layers.get_output(network),1e-7,1.0-1.e-7)
    
    # Trained weights of the network
    if case_study==1:
        params_file="2d_ti/vaecnn2d_100ep_unc.pkl"
    else:
        params_file="2d_ti/vaecnn2d_100ep_con.pkl"

if case_study == 3 or  case_study == 4:
    print(str(case_study))
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
    
    # Trained weights of the network
    if case_study==3:
        params_file="3d_ti/vaecnn3d_100ep_unc.pkl"
    else:
        params_file="3d_ti/vaecnn3d_100ep_con.pkl"

# Load trained weights of the network
saved_models_dir="/home/elaloy/VAE_MCMC/saved_models/"
with open(saved_models_dir+params_file,'rb') as f:
    updated_param_values=pickle.load(f)
    lasagne.layers.set_all_param_values(network, updated_param_values)
#%% Check predicton at testing time
output = lasagne.layers.get_output(network)
ae_encdec = theano.function([input_var], output) #create encoding-decoding function

if case_study == 1 or  case_study == 2:    
    X_test_pred = ae_encdec(X_test[:1000]).reshape(-1, 100, 100)
    X_test_pred[X_test_pred>=0.5]=1
    X_test_pred[X_test_pred<0.5]=0

if case_study == 3 or  case_study == 4:
   
    X_test_pred = ae_encdec(X_test[:1000]).reshape(-1, 27, 32, 30)
    X_test_pred[X_test_pred<0.5]=0
    X_test_pred[X_test_pred>=0.5]=1

# Quick and dirty plot
if case_study < 3:  
    plt.imshow(X_test_pred[-1,:,:])
else:
    plt.imshow(X_test_pred[-1,:,:,15])

#%% Verify visually the normality of encoded latent variables
encoded = lasagne.layers.get_output(q_layer)
ae_encode = theano.function([input_var], encoded)
#X_encoded = ae_encode(X[:10000])
start_time = time.time()
X_encoded=np.zeros((X.shape[0],z_hid))
bsize=1000
ns=5000
X_encoded=np.zeros((ns,z_hid))
nb=2
for i in range(0,ns/bsize):
    i_start=0+(i-1)*bsize
    i_end=i*bsize-1
    X_encoded[i_start:i_end,:] = ae_encode(X[i_start:i_end,:,:,:])
end_time = time.time()
elapsed_time = end_time - start_time
print("Time encoding training data = %5.4f seconds." % (elapsed_time))
print(X_encoded.shape)

# Randomly choose dimensions and quick and dirty plot
index1 = np.random.randint(z_hid)
index2 = np.random.randint(z_hid)
plt.figure(figsize=(6,6))
plt.scatter(X_encoded[:,index1],X_encoded[:,index2], alpha = 0.03)
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.show()
#%% Check VAE-based realizations
# First build the generator
noise_var = T.matrix('noise_inputs')
gen = get_output(network, {q_layer:noise_var})
generate_fn = theano.function([noise_var], gen)

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
#% Second set dimensions
try:
  del X1, X2, X3, X_gen_raw, X_gen_loop, generated_i
except: 
  pass
if case_study == 1 or  case_study == 2:
    n1=100
    n2=100
    n3=1
    if case_study==2: # conditioning points
        idx=np.array([ [20,20], [35,35], [65,65], [80,80], [20,80], [80,20], [35,65], [65,35], [51,51]])-1
if case_study == 5 or  case_study == 6: 
    n1=32
    n2=30
    n3=27
# Third generate the realizations     
np.random.seed(2047)
nrz=10
WriteToDisk=False
noise_input = np.random.randn(nrz,z_hid).astype('float32')
t1=time.time()
generated_i = generate_fn(noise_input)
t2=time.time()
X_gen_raw=np.array(generated_i).reshape(-1, n3, n1, n2)
nloop=10
generated = generated_i.reshape(-1, n3, n1, n2)
for ii in range(0,nloop):
     mu=ae_encode_mu(generated)
     log_sigma=ae_encode_log_sigma(generated)
     noise_adj=noise_adjust(noise_input,mu,log_sigma)
     generated = gen_from_noise(noise_adj)
     generated = generated.reshape(-1, n3, n1, n2)
X_gen_loop = generated
t3=time.time()
print((t2-t1)) 
print((t3-t1))

#thresholding
if case_study == 1 or  case_study == 2:
    X_gen_raw = np.rint(X_gen_raw).astype(int)
    X_gen_loop = np.rint(X_gen_loop).astype(int)

if case_study == 3 or case_study == 4:
    X_gen_raw[X_gen_raw>=0.5]=1
    X_gen_raw[X_gen_raw<0.5]=0
    X_gen_loop[X_gen_loop>=0.5]=1
    X_gen_loop[X_gen_loop<0.5]=0
    
# Save to disk
if WriteToDisk==True:
    if case_study==1 or case_study==3:
        h5_filename="gen_unc_"+str(case_study)+'_'+str(nrz)
    else:
        h5_filename="gen_con_"+str(case_study)+'_'+str(nrz)
    f = h5py.File(h5_filename+'.hdf5', mode='w')
    h5dset = f.create_dataset('features', data=X_gen_loop)
    f.flush()
    f.close()

#%% Plot the 2D realizations - (3D realizations have herein been plotted with 
# Matlab using the hdf realizations written to disk)

savefigure=False
sub_letter=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)',
            '(k)','(l)','(m)']
colormap=['gray','gray','gray','gray','gray','gray']

from matplotlib import rc
rc('text', usetex=True)
if case_study < 3:

    show_no_loop=False
    X1=np.array(X[0:4,])
    X2=np.array(X_gen_raw[0:8,])
    X3=np.array(X_gen_loop[0:8,])
    
    
    filename='fig_rnd_realz_case_'+str(case_study)
    xstep=np.round(n1/5.0).astype('int')
    xx=np.arange(0,n1+xstep,xstep)
    xx[-1]=n1-1
    xlabel=xx.tolist()
    xlabel[0]='1'
    xlabel[-1]=str(n1)
    fig=plt.figure(figsize=(10,8.5))

    tt=0
    for i in xrange(0,12):
        tt=tt+1
        sub=plt.subplot(3,4,tt)
        sub.axes.tick_params(labelsize=10)
     
        if tt==9:
            sub.axes.get_xaxis().set_ticks(xx)
            sub.axes.get_yaxis().set_ticks(xx)
            sub.axes.set_xticklabels(xlabel)
            sub.axes.set_yticklabels(xlabel)
            sub.axes.set_ylabel(r'\textit{y} [m]', fontsize = 12.0)
            sub.axes.set_xlabel(r'\textit{x} [m]', fontsize = 12.0)
        elif tt==1 or tt==5:
            sub.axes.get_xaxis().set_ticks([])
            sub.axes.get_yaxis().set_ticks(xx)
            sub.axes.set_ylabel(r'\textit{y} [m]', fontsize = 12.0)
            sub.axes.set_yticklabels(xlabel)
        elif tt>9:
            sub.axes.get_yaxis().set_ticks([])
            sub.axes.get_xaxis().set_ticks(xx)
            sub.axes.set_xlabel(r'\textit{x} [m]', fontsize = 12.0)
            sub.axes.set_xticklabels(xlabel)
        else:
            sub.axes.get_xaxis().set_ticks([])
            sub.axes.get_yaxis().set_ticks([])
        if tt < 5:
            plt.imshow(np.reshape(X1[i,:,:,:],(n1,n2)),cmap=colormap[case_study-1])
        elif tt < 9:
            if show_no_loop==True:
                plt.imshow(np.reshape(X2[i-4,:,:,:],(n1,n2)),colormap[case_study-1])
            else:
                plt.imshow(np.reshape(X3[i-4,:,:,:],(n1,n2)),colormap[case_study-1])
        else:
            if show_no_loop==True:
                plt.imshow(np.reshape(X2[i-4,:,:,:],(n1,n2)),colormap[case_study-1])
            else:
                plt.imshow(np.reshape(X3[i-4,:,:,:],(n1,n2)),colormap[case_study-1])
        if i>=0 and case_study==2:
            plt.hold(True)
            plt.plot(idx[:,0],idx[:,1], "o", markeredgewidth=0.5, markersize=8,markeredgecolor='r',markerfacecolor='none')
        if i < 4:
            str_title=sub_letter[i]+' Training \#'+'%1.0f' % (i+1)
        elif i < 8:
            if show_no_loop==True:
                str_title=' DR-0 \#'+'%1.0f' % (i-4+1)
            else:
                str_title=sub_letter[i]+' DR \#'+'%1.0f' % (i-4+1)
        else:
            if show_no_loop==True:
                str_title='DR-10 \#'+'%1.0f' % (i-8+1)
            else:            
                str_title=sub_letter[i]+' DR \#'+'%1.0f' % (i-4+1)#+
        sub.set_title(str_title,size=14)
            
    if savefigure==True:
        fig.savefig(filename+'.pdf',dpi=600)
elif case_study > 3: # save realizations for 3D plotting with another software
    if case_study==3:
        ctype="unc"
    else:
        ctype="con"
    h5_filename="3d_test_"+ctype
    f = h5py.File(h5_filename+'.hdf5', mode='w')
    h5dset = f.create_dataset('features', data=X_test)
    f.flush()
    f.close()
    
    h5_filename="3d_gen_raw"+ctype+"_%1.0" % (z_hid)
    f = h5py.File(h5_filename+'.hdf5', mode='w')
    h5dset = f.create_dataset('features', data=X_gen_raw)
    f.flush()
    f.close()
    
    h5_filename="3d_gen_loop_"+ctype+"_%1.0" % (z_hid)
    f = h5py.File(h5_filename+'.hdf5', mode='w')
    h5dset = f.create_dataset('features', data=X_gen_loop)
    f.flush()
    f.close()
 
#%% Lastly check conditioning accuracy for case studies 2 and 4
if case_study==2:
    ns=1000
    np.random.seed(2047)
    noise_input = np.random.randn(ns,z_hid).astype('float32')
    generated_i = generate_fn(noise_input)
    nloop=10
    generated = generated_i.reshape(-1, 1, 100, 100)
    X_gen_raw = generated_i.reshape(-1, 1, 100, 100)
    for ii in range(0,nloop):
         mu=ae_encode_mu(generated)
         log_sigma=ae_encode_log_sigma(generated)
         noise_adj=noise_adjust(noise_input,mu,log_sigma)
         generated = gen_from_noise(noise_adj)
         generated = generated.reshape(-1, 1, 100, 100)
    X_gen_loop = generated#* 255
    X_gen_loop[X_gen_loop>=0.5]=1
    X_gen_loop[X_gen_loop<0.5]=0
    
    X_gen_raw[X_gen_raw>=0.5]=1
    X_gen_raw[X_gen_raw<0.5]=0
    
    # First check training set
    cdt=np.zeros((X.shape[0],9))-999
    idx=np.array([ [20,20], [35,35], [65,65], [80,80], [20,80], [80,20], [35,65], [65,35], [51,51]])-1
    for i in range(0,X.shape[0]):
        for j in range(0,9):
            cdt[i,j]=X[i,0,idx[j,0],idx[j,1]]
            
    ii=np.where(np.sum(cdt,axis=1) != 3)
    perci=1-len(ii[0])/1000.0
    print(perci) 
    ii=np.logical_or(np.sum(cdt,axis=1) > 4, np.sum(cdt,axis=1) < 2)
    perci=1-len(ii[ii==True])/1000.0
    print(perci) 
    
    # Then the encoded-decoded models from the testing set
    X_pred=X_test_pred
    cdp=np.zeros((X_pred.shape[0],9))-999
    for i in range(0,X_pred.shape[0]):
        for j in range(0,9):
            cdp[i,j]=X_pred[i,idx[j,0],idx[j,1]]
            
    # And finally X_gen_raw and  X_gen_loop
    X_gen_=np.array(X_gen_raw)
    cdg0=np.zeros((X_gen_.shape[0],9))-999
    for i in range(0,X_gen_.shape[0]):
        img=np.reshape(X_gen_[i,:],(100,100))
        for j in range(0,9):
            cdg0[i,j]=img[idx[j,0],idx[j,1]]
            
    X_gen_=np.array(X_gen_loop)
    cdg=np.zeros((X_gen_.shape[0],9))-999
    for i in range(0,X_gen_.shape[0]):
        img=np.reshape(X_gen_[i,:],(100,100))
        for j in range(0,9):
            cdg[i,j]=img[idx[j,0],idx[j,1]]
    
    print(np.sum(cdt,axis=0)/cdt.shape[0])
    print(np.sum(cdp,axis=0)/cdp.shape[0])
    print(np.sum(cdg,axis=0)/cdg.shape[0])
    
    ii=np.where(np.sum(cdg0,axis=1) != 3)
    perc0=1-len(ii[0])/1000.0
    print(perc0)
    ii=np.logical_or(np.sum(cdg0,axis=1) > 4, np.sum(cdg0,axis=1) < 2)
    perc0=1-len(ii[ii==True])/1000.0
    print(perc0)
    
    ii=np.where(np.sum(cdg,axis=1) != 3)
    perc=1-len(ii[0])/1000.0
    print(perc)
    ii=np.logical_or(np.sum(cdg,axis=1) > 4, np.sum(cdg,axis=1) < 2)
    perc=1-len(ii[ii==True])/1000.0
    print(perc)
   
    ceff=np.maximum(np.mean(cdg,axis=0),1-np.mean(cdg,axis=0))
    print(ceff)
    idx1=np.array([0,4,7],dtype=np.int)
    idx0=np.array([1,2,3,5,6,8],dtype=np.int)
    print(np.mean(ceff[idx1]))
    print(np.mean(ceff[idx0]))
#%%
if case_study == 4:
    import scipy.io as sio
    current_dir=os.getcwd()
    idx_dir='/home/elaloy/train_data/3D_hydraulic_tomo'
    os.chdir(idx_dir)
    idx_dic = sio.loadmat('coord_obs_4m.mat')
    
#    idx=idx_dic['coord_obs']-1
#    kt_dic=sio.loadmat('trueK_con.mat')
#    kt=kt_dic['K']
#    kt= np.float32(np.swapaxes(kt, 0, 2)) #
    
    os.chdir(idx_dir)
    os.chdir('/home/elaloy/train_data/CGC_3D')
    kv_dic=sio.loadmat('kv.mat')
    kv=kv_dic['kv']
    kv=kv[:,3]
    os.chdir(current_dir)
    
#    cdi=np.zeros((1,56))-999
#    for j in range(0,56):
#        cdi[0,j]=np.abs(kt[idx[j,0],idx[j,1],idx[j,2]]-kv[j])
    
    # First check training set
    X_=np.float32(X)
    cdt=np.zeros((X_.shape[0],56))-999
    cdti=np.zeros((X_.shape[0],56))-999
    for i in range(0,X_.shape[0]):
        for j in range(0,56):
            cdt[i,j]=np.abs(X_[i,idx[j,0],idx[j,1],idx[j,2]]-kv[j])
            cdti[i,j]=X_[i,idx[j,0],idx[j,1],idx[j,2]]
    
    ii=np.where(np.sum(cdt,axis=1) != 0)
    perci=1-len(ii[0])/np.float(cdt.shape[0])
    print(perci) 
    ii=np.where(np.sum(cdt,axis=1) > 10)
    perci=1-len(ii[0])/np.float(cdt.shape[0])
    print(perci) 
    
    # Then the encoded-decoded models from the testing set
    X_test=np.float32(X_test_pred)
    cdp=np.zeros((X_test.shape[0],56))-999
    for i in range(0,X_test.shape[0]):
        for j in range(0,56):
            cdp[i,j]=np.abs(X_test[i,idx[j,0],idx[j,1],idx[j,2]]-kv[j])
    print(np.sum(cdp,axis=1))
    
    # And finally X_gen_raw and  X_gen_loop
    X_gen_=np.array(X_gen_raw)
    cdg0=np.zeros((X_gen_.shape[0],56))-999
    for i in range(0,X_gen_.shape[0]):
        for j in range(0,56):
            cdg0[i,j]=np.abs(X_gen_[i,idx[j,0],idx[j,1],idx[j,2]]-kv[j])
            
    X_gen_=np.array(X_gen_loop)
    cdg=np.zeros((X_gen_.shape[0],56))-999
    for i in range(0,X_gen_.shape[0]):
        for j in range(0,56):
            cdg[i,j]=np.abs(X_gen_[i,idx[j,0],idx[j,1],idx[j,2]]-kv[j])
            
    ii=np.where(np.sum(cdg0,axis=1) != 0)
    perc0=1-len(ii[0])/np.float(cdg0.shape[0])
    print(perc0)
    ii=np.where(np.sum(cdg0,axis=1) > 10)
    perc0=1-len(ii[0])/np.float(cdg0.shape[0])
    print(perc0)
    
    ii=np.where(np.sum(cdg,axis=1) != 0)
    perc=1-len(ii[0])/np.float(cdg.shape[0])
    print(perc)
    ii=np.where(np.sum(cdg,axis=1) > 10)
    perc=1-len(ii[0])/np.float(cdg.shape[0])
    print(perc)
    
    ceff0=np.maximum(np.mean(cdt,axis=0),1-np.mean(cdt,axis=0))
    ceff=np.maximum(np.mean(cdg,axis=0),1-np.mean(cdg,axis=0))
    print(ceff)
