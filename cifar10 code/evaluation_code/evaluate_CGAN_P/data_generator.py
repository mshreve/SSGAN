#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 16:01:29 2017

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
import PIL

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from theano.tensor.extra_ops import diff
import lasagne.layers as ll


from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch

from keras.datasets import cifar10

def transform(X):
    X = [center_crop(x, npx) for x in X]
    return floatX(X).transpose(0, 3, 1, 2)/127.5 - 1.

def inverse_transform(X):
    X = (X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1)+1.)/2.
    return X
    
k = 1             # # of discrim updates for each gen update
l2 = 1e-5         # l2 weight decay
nv = 14;nvis = 196# # of samples to visualize during training
b1 = 0.5          # momentum term of adam
nc = 3            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 32          # # of pixels width/height of images
nz = 100          # # of dim for Z
ny = 10          # # of dim for Y
ngf = 128         # # of gen filters in first conv layer
ndf = 64         # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 75        # # of iter at starting learning rate
niter_decay = 0   # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
ntrain = 350000   # # of examples to train on
eps = 10**-20     # # to avoid division of zero errors
attr_prob = 0.15  # # attribute probability
nlab = 4000       # # number of labeled examples
alpha = 1.0
offset = 00000


vaX, vaY = cifar10.load_data()[0]
#vaX = []
#for v in vaX32:
#    im = PIL.Image.fromarray(v)
#    vaX.append(np.array(im.resize((npx,npx), PIL.Image.BICUBIC)))
    
#vaX = floatX(vaX)[:,:,:,np.newaxis]
vaX = floatX(vaX).transpose(0, 3, 1, 2)/127.5 - 1.
ntrain = len(vaX)
vaY = np.array(vaY)
ntrain = ntrain/nbatch * nbatch
vaX = vaX[:ntrain]; vaY = vaY[:ntrain]
vaX, vaY = shuffle(vaX, vaY); vaX, vaY = shuffle(vaX, vaY)

ndig = nlab/ny
vlist = []
for i in range(ny):
    vlist += list(np.where(vaY==i)[0][:ndig])

vaY = floatX(OneHot(vaY,ny))
laX = vaX[vlist]; laY=vaY[vlist]
sample_vec = np.random.choice(range(len(laX)),len(vaX),replace=True)
laX = laX[sample_vec]; laY = laY[sample_vec]

taX, taY = cifar10.load_data()[1]
#taX = []
#for v in taX32:
#    im = PIL.Image.fromarray(v)
#    taX.append(np.array(im.resize((npx,npx), PIL.Image.BICUBIC)))
#taX = floatX(taX)[:,:,:,np.newaxis]
taX = floatX(taX).transpose(0, 3, 1, 2)/255.0 - 0.
ntest = len(taX)
taY = np.array(taY)
taY = floatX(OneHot(taY,ny))
ntest = ntest/nbatch * nbatch
taX=taX[:ntest]; taY=taY[:ntest]

   
relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy
cce = T.nnet.categorical_crossentropy
softmax = activations.Softmax()

def gen(Z, Y, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wx):
    Z = T.concatenate([Z, Y], axis=1)
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf*8*2, 2, 2))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = relu(batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    x = tanh(deconv(h4, wx, subsample=(2, 2), border_mode=(2, 2)))      
    return x
    
desc_full = 'SS_S_GAN' # use 'SS_C_GAN', 'SS_AC_GAN', 'C_GAN' and 'AC_GAN' to produce samples from other models 

X = T.tensor4(); Z = T.matrix(); Y = T.matrix()

ss_gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(desc_full))]
ss_gX = gen(Z, Y, *ss_gen_params)
ss_gen = theano.function([Z, Y], ss_gX, on_unused_input='ignore')

nvis=ntrain

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(nvis, nz)))
sample_ymb = floatX(vaY[:nvis])


smb = []; ymb = []
for idx in tqdm(xrange(0, ntrain/nbatch), total=ntrain/nbatch):
    smb.append(np.asarray(ss_gen(sample_zmb[idx*nbatch:((idx+1)*nbatch)], vaY[idx*nbatch:((idx+1)*nbatch)])))
    ymb.append(vaY[idx*nbatch:((idx+1)*nbatch)])  
    
smb = np.concatenate(smb)
ymb = np.concatenate(ymb)
smb = (smb+1)*127.5
laX = (laX+1)*127.5

x = []; y = []
sample_vec = np.random.choice(range(len(laX)),25000,replace=False)
x.append(smb[sample_vec]); y.append(ymb[sample_vec])
sample_vec = np.random.choice(range(len(laX)),25000,replace=False)
x.append(laX[sample_vec]); y.append(laY[sample_vec])
x = np.concatenate(x); y = np.concatenate(y)
x, y = shuffle(x, y)

xx = x.transpose(0,2,3,1).reshape((x.shape[0],32**2,3))
xxx = xx.reshape((x.shape[0],32**2*3),order='F')
y = np.argmax(y,axis=1)[:,np.newaxis]
y = [yy[0] for yy in y]

import cPickle
for i in range(1):
    f = open('cifar-10-batches-py/ss_syn_data_batch_'+`i+1`+'_'+`nlab`, 'wb')
    f = open('cifar-10-batches-py/ss_syn_test_batch'+'_'+`nlab`, 'wb')
    dd = {}
    dd['data'] = xxx[(10000*i):(10000*(i+1))]
    dd['labels'] = y[(10000*i):(10000*(i+1))]
    cPickle.dump(dd, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    



