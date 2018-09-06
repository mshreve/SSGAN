#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:49:01 2017

@author: skumar
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:01:32 2017

@author: skumar

code for evaluating GAN's - create a network to measure similarity to labeled samples
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

from load import faces

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
npx = 64          # # of pixels width/height of images
nz = 100          # # of dim for Z
ny = 18           # # of dim for Y
ngf = 128         # # of gen filters in first conv layer
ndf = 128         # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 05        # # of iter at starting learning rate
niter_decay = 0   # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
ntrain = 350000   # # of examples to train on
eps = 10**-20     # # to avoid division of zero errors
attr_prob = 0.15  # # attribute probability
offset=0
nlab = 1000

nlab = nlab/nbatch*nbatch
vaX, vaY = faces()
ntrain = len(vaX)
vaX = transform(vaX)

laX = vaX[:nlab]; laY = vaY[:nlab]

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy
cce = T.nnet.categorical_crossentropy
X = T.tensor4(); Z = T.matrix(); Y = T.matrix()

def gen(Z, Y, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wx):
    Z = T.concatenate([Z, Y], axis=1)
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf*8, 4, 4))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = relu(batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    x = tanh(deconv(h4, wx, subsample=(2, 2), border_mode=(2, 2)))      
    return x


desc_full = 'AC_GAN'
desc_full = 'SS_AC_GAN'
desc_full = 'SS_S_GAN'
desc_full = 'C_GAN'
desc_full = 'SS_C_GAN'

desc_full = 'AC_GAN'
am_gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(desc_full))]
am_gX = gen(Z, Y, *am_gen_params)
am_gen = theano.function([Z, Y], am_gX, on_unused_input='ignore')

desc_full = 'SS_AC_GAN'
gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(desc_full))]
ac_gX = gen(Z, Y, *gen_params)
ac_gen = theano.function([Z, Y], ac_gX, on_unused_input='ignore')

desc_full = 'SS_S_GAN'
gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(desc_full))]
ss_gX = gen(Z, Y, *gen_params)
ss_gen = theano.function([Z, Y], ss_gX, on_unused_input='ignore')

desc_full = 'C_GAN'
gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(desc_full))]
s_gX_small = gen(Z, Y, *gen_params)
s_gen_small = theano.function([Z, Y], s_gX_small, on_unused_input='ignore')

desc_full = 'SS_C_GAN'
gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(desc_full))]
s_gX = gen(Z, Y, *gen_params)
s_gen = theano.function([Z, Y], s_gX, on_unused_input='ignore')



#desc_full = 'sup_dcgan_1000'
#gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(desc_full))]
#$s_gX = gen(Z, Y, *gen_params)

def discrim(X, w, w2, g2, b2, w3, g3, b3, w4, g4, b4, wy):
    #yb = Y.dimshuffle(0, 1, 'x', 'x')
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    #h = conv_cond_concat(h, yb)
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))        
    h4 = T.flatten(h4, 2)
    p = sigmoid(T.dot(h4,wy)) # p = prob. of image being real or fake
    return p
desc_full = 'sample_predict'
discrim_params = [sharedX(p) for p in joblib.load('models/%s/discrim_params.jl'%(desc_full))]
dX = discrim(X, *discrim_params)
dx = theano.function([X], dX, on_unused_input='ignore')


nvis=10000
offset = 16000

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(nvis, nz)))
sample_ymb = floatX(vaY[offset:offset+nvis])

true_samples = vaX[:128]
fake_samples = vaX[offset:offset+nvis]
ac_samples = np.asarray(ac_gen(sample_zmb, sample_ymb))
am_samples = np.asarray(am_gen(sample_zmb, sample_ymb))
ss_samples = np.asarray(ss_gen(sample_zmb, sample_ymb))
s_samples_small = np.asarray(s_gen_small(sample_zmb, sample_ymb))
s_samples = np.asarray(s_gen(sample_zmb, sample_ymb))

true_pred = dx(true_samples)
fake_pred = dx(fake_samples)
ac_pred = dx(ac_samples)
am_pred = dx(am_samples)
ss_pred = dx(ss_samples)
s_pred_small = dx(s_samples_small)
s_pred = dx(s_samples)

print np.mean(true_pred)#>0.5)
print np.mean(fake_pred)#>0.5)
print np.mean(ac_pred)#>0.5)
print np.mean(am_pred)#>0.5)
print np.mean(ss_pred)#>0.5)
print np.mean(s_pred_small)#>0.5)
print np.mean(s_pred)#>0.5)
