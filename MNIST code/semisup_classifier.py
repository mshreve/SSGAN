#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 19:07:09 2017

@author: skumar
"""

import sys
sys.path.append('..')

import os
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from theano.tensor.extra_ops import diff


from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch

from keras.datasets import mnist

def transform(X):
    X = [center_crop(x, npx) for x in X]
    return floatX(X).transpose(0, 3, 1, 2)/255.0 - 0.

def inverse_transform(X):
    X = (X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1)+0.)/1.
    return X
    
k = 1             # # of discrim updates for each gen update
l2 = 2.5e-5       # l2 weight decay
nv = 30;nvis = 900# # of samples to visualize during training
b1 = 0.5          # momentum term of adam
nc = 1            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 28          # # of pixels width/height of images
nz = 25           # # of dim for Z
ny = 10           # # of dim for Y
ngf = 64          # # of gen filters in first conv layer
ndf = 64          # # of discrim filters in first conv layer
ngfc = 1024       # # of gen units for fully connected layers
ndfc = 1024       # # of discrim units for fully connected layers
nx = npx*npx*nc   # # of dimensions in X
niter = 150       # # of iter at starting learning rate
niter_decay = 0   # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
ntrain = 350000   # # of examples to train on
eps = 10**-20     # # to avoid division of zero errors
attr_prob = 0.15  # # attribute probability
nlab = 20         # # number of labeled examples
alpha = 1.0
offset=0

vaX, vaY = mnist.load_data()[0]
vaX = floatX(vaX)[:,:,:,np.newaxis]
vaX = vaX.transpose(0, 3, 1, 2)/255.0 - 0.
ntrain = len(vaX)
vaY = np.array(vaY)
ntrain = ntrain/nbatch * nbatch
vaX=vaX[:ntrain]; vaY=vaY[:ntrain]

ss = 1 #seed controls which digits are chosen as labeled samples
for _ in range(ss):
    vaX, vaY = shuffle(vaX, vaY); 
ndig = nlab/ny
vlist = []
for i in range(ny):
    vlist += list(np.where(vaY==i)[0][:ndig])
vaY = floatX(OneHot(vaY,ny))
laX = vaX[vlist]; laY=vaY[vlist]
sample_vec = np.random.choice(range(len(laX)),len(vaX),replace=True)
laX = laX[sample_vec]; laY = laY[sample_vec]

taX, taY = mnist.load_data()[1]
taX = floatX(taX)[:,:,:,np.newaxis]
taX = taX.transpose(0, 3, 1, 2)/255.0 - 0.
ntest = len(taX)
taY = np.array(taY)
taY = floatX(OneHot(taY,ny))
ntest = ntest/nbatch * nbatch
taX=taX[:ntest]; taY=taY[:ntest]

desc = 'mnist_semisup_predict'
desc_full = 'mnist_semisup_predict'
model_dir = 'models/%s'%desc_full
samples_dir = 'samples/%s'%desc
if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)
    
relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy
cce = T.nnet.categorical_crossentropy
softmax = activations.Softmax()

difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)

dw  = difn((ndf, nc+0*ny, 5, 5), 'dw'); dw2 = difn((ndf*2, ndf, 5, 5), 'dw2')
dw3 = difn((ndf*2*7*7, ndfc), 'dw3'); dwy = difn((ndfc, ny), 'dwy')

discrim_params = [dw, dw2, dw3, dwy]

def discrim(X, w, w2, w3, wy):    
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))))
    h2 = T.flatten(h2, 2)
    h3 = lrelu(batchnorm(T.dot(h2, w3)))
    q = softmax(T.dot(h3, wy))
    return q

X = T.tensor4(); Z = T.matrix(); Y = T.matrix(); W = T.tensor4();

def gen(Z, Y, w, w2, w3, wx):
    Z = T.concatenate([Z, Y], axis=1)
    h = relu(batchnorm(T.dot(Z, w)))
    h2 = relu(batchnorm(T.dot(h, w2)))
    h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
    x = sigmoid(deconv(h3, wx, subsample=(2, 2), border_mode=(2, 2)))
    return x
    
desc_full = 'mnist_SS_Sgan_'+str(nlab)+'_'+str(alpha)+'_'+str(1)+'_'+str(1) #just use one model for illustration
    

ss_gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(desc_full))]
ss_gX = gen(Z, Y, *ss_gen_params)
ss_gen = theano.function([Z, Y], ss_gX, on_unused_input='ignore')

nvis=ntrain

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(nvis, nz)))
sample_ymb = floatX(vaY[:nvis])

Yhat = discrim(X, *discrim_params)
Ytild = discrim(W, *discrim_params)

cost = cce(Yhat,Y).mean() 
cost2 = T.mean(T.sum(-Ytild*T.log(Ytild+eps),axis=1))
cost += 1*cost2

lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
d_updates = d_updater(discrim_params, cost)
d_updates2 = d_updater(discrim_params, cost2)

print 'COMPILING'
t = time()
_train_d = theano.function([X, Y, W], cost, updates=d_updates, on_unused_input='ignore')
_train_d2 = theano.function([W], cost2, updates=d_updates2, on_unused_input='ignore')

dx = theano.function([X],Yhat)
print '%.2f seconds to compile theano functions'%(time()-t)


print desc.upper()
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()
niter = 20
for epoch in range(niter):
    vaX, vaY, sample_zmb = shuffle(vaX, vaY, sample_zmb); #sample_zmb, sample_ymb = shuffle()
    for idx in tqdm(xrange(0, ntrain/nbatch), total=ntrain/nbatch):
        imb = vaX[idx*nbatch:((idx+1)*nbatch)]
        ymb = floatX(vaY[idx*nbatch:((idx+1)*nbatch)])
        smb = np.asarray(ss_gen(sample_zmb[idx*nbatch:((idx+1)*nbatch)], vaY[idx*nbatch:((idx+1)*nbatch)]))
        
        #labeled set
        l_imb = laX[idx*nbatch:((idx+1)*nbatch)]
        l_ymb = floatX(laY[idx*nbatch:((idx+1)*nbatch)])
        
        smb = smb[:nbatch]; l_imb = np.concatenate([smb, l_imb])
        ymb = ymb[:nbatch]; l_ymb = np.concatenate([ymb, l_ymb])
         
        #a_cost = _train_d(smb, ymb) #train over synthetic examples
        a_cost = _train_d(l_imb, l_ymb, imb) #train over (small number of) labeled examples
        #a_cost = _train_d2(imb)
        n_updates += 1
        n_examples += len(imb)
    
    print "Epoch =", epoch
    print "Cost vec =", np.asarray(a_cost)

        
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
        
    joblib.dump([p.get_value() for p in discrim_params], 'models/%s/discrim_params.jl'%(desc_full))
    
taX, taY = shuffle(taX, taY)
err = 0
for idx in tqdm(xrange(0, ntest/nbatch), total=ntest/nbatch):
    imb = taX[idx*nbatch:((idx+1)*nbatch)]
    ymb = floatX(taY[idx*nbatch:((idx+1)*nbatch)])
    ymbhat = dx(imb)
    
    y_true = np.argmax(ymb,axis=1)
    y_hat = np.argmax(ymbhat,axis=1)
    err += len(np.nonzero(y_true-y_hat)[0])    
    
print err # this is the classification error using the samples generated by SS_S_GAN
    
    
    