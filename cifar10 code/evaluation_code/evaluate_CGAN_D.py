#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:20:56 2017

@author: skumar

code for evaluating GAN's - extract discriminator features and evaluate. 
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
ny = 10           # # of dim for Y
ngf = 128         # # of gen filters in first conv layer
ndf = 64         # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 149        # # of iter at starting learning rate
niter_decay = 0   # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
ntrain = 350000   # # of examples to train on
eps = 10**-20     # # to avoid division of zero errors
attr_prob = 0.15  # # attribute probability
nlab = 4000       # # number of labeled examples
alpha = 1.0
offset = 00000

vaX, vaY = cifar10.load_data()[0]
vaX = floatX(vaX).transpose(0, 3, 1, 2)/127.5 - 1.
ntrain = len(vaX)
vaY = np.array(vaY)
VAY = np.array(vaY).reshape(ntrain,)
ntrain = ntrain/nbatch * nbatch
vaX = vaX[:ntrain]; vaY = vaY[:ntrain]
vaX, vaY = shuffle(vaX, vaY); vaX, vaY = shuffle(vaX, vaY)

ndig = nlab/ny
vlist = []
for i in range(ny):
    vlist += list(np.where(vaY==i)[0][:ndig])

vaY = floatX(OneHot(vaY,ny))

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy
cce = T.nnet.categorical_crossentropy
X = T.tensor4(); Z = T.matrix(); Y = T.matrix()

def s_discrim(X, Y, w, w2, g2, b2, w3, g3, b3, w4, g4, b4, wy):
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    h = conv_cond_concat(h, yb)
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))        
    h4f = T.flatten(h4, 2)
    return h4f

def ac_discrim(X, Y, w, w2, g2, b2, w3, g3, b3, w4, g4, b4, wy, wq):
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    #h = conv_cond_concat(h, yb)
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))        
    h4f = T.flatten(h4, 2)
    return h4f

def ss_discrim(X, Y, w, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5, wy, wq):
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    #h = conv_cond_concat(h, yb)
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))        
    h4f = T.flatten(h4, 2)
    return h4f

desc_full = 'AC_GAN'
disc_params = [sharedX(p) for p in joblib.load('models/%s/discrim_params.jl'%(desc_full))]
am_dX = ac_discrim(X, Y, *disc_params)
am_disc = theano.function([X, Y], am_dX, on_unused_input='ignore')

desc_full = 'SS_AC_GAN'
disc_params = [sharedX(p) for p in joblib.load('models/%s/discrim_params.jl'%(desc_full))]
ac_dX = ac_discrim(X, Y, *disc_params)
ac_disc = theano.function([X, Y], ac_dX, on_unused_input='ignore')

desc_full = 'SS_S_GAN'
disc_params = [sharedX(p) for p in joblib.load('models/%s/discrim_params.jl'%(desc_full))]
ss_dX = ss_discrim(X, Y, *disc_params)
ss_disc = theano.function([X, Y], ss_dX, on_unused_input='ignore')

desc_full = 'SS_C_GAN'
disc_params = [sharedX(p) for p in joblib.load('models/%s/discrim_params.jl'%(desc_full))]
s_dX_small = s_discrim(X, Y, *disc_params)
s_disc_small = theano.function([X, Y], s_dX_small, on_unused_input='ignore')

desc_full = 'C_GAN'
disc_params = [sharedX(p) for p in joblib.load('models/%s/discrim_params.jl'%(desc_full))]
s_dX = s_discrim(X, Y, *disc_params)
s_disc = theano.function([X, Y], s_dX, on_unused_input='ignore')

nvis=1000
offset = 10000
sample_zmb = floatX(np_rng.uniform(-1., 1., size=(nvis, nz)))
sample_ymb = floatX(vaY[offset:offset+nvis])

sample_ymb_C = VAY[offset:offset+nvis]

sample_xmb = vaX[offset:offset+nvis]

am_feat = np.asarray(am_disc(sample_xmb, sample_ymb))
ac_feat = np.asarray(ac_disc(sample_xmb, sample_ymb))
ss_feat = np.asarray(ss_disc(sample_xmb, sample_ymb))
sm_feat = np.asarray(s_disc_small(sample_xmb, sample_ymb))
s_feat = np.asarray(s_disc(sample_xmb, sample_ymb))

from sklearn import linear_model, metrics, svm
from sklearn.model_selection import cross_val_predict

#logreg = linear_model.LogisticRegression(C=1e5)
logreg =  svm.SVC(gamma=0.001)

am_pred = metrics.accuracy_score(sample_ymb_C,cross_val_predict(logreg, am_feat, sample_ymb_C, cv=10))
ac_pred = metrics.accuracy_score(sample_ymb_C,cross_val_predict(logreg, ac_feat, sample_ymb_C, cv=10))
ss_pred = metrics.accuracy_score(sample_ymb_C,cross_val_predict(logreg, ss_feat, sample_ymb_C, cv=10))
sm_pred = metrics.accuracy_score(sample_ymb_C,cross_val_predict(logreg, sm_feat, sample_ymb_C, cv=10))
s_pred = metrics.accuracy_score(sample_ymb_C,cross_val_predict(logreg, s_feat, sample_ymb_C, cv=10))

