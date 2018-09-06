#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 13:17:44 2017

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
niter = 75        # # of iter at starting learning rate
niter_decay = 0   # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
ntrain = 350000   # # of examples to train on
eps = 10**-20     # # to avoid division of zero errors
attr_prob = 0.15  # # attribute probability
nlab = 1000       # # number of labeled examples

nlab = nlab/nbatch*nbatch
vaX, vaY = faces()
ntrain = len(vaX)
vaX = transform(vaX)


desc = 'attr_predict'
desc_full = 'attr_predict'
model_dir = 'models/%s'%desc
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

difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)

dw  = difn((ndf, nc, 5, 5), 'dw'); dw2 = difn((ndf*2, ndf+0*ny, 5, 5), 'dw2'); dg2 = gain_ifn((ndf*2), 'dg2')
db2 = bias_ifn((ndf*2), 'db2'); dw3 = difn((ndf*4, ndf*2, 5, 5), 'dw3'); dg3 = gain_ifn((ndf*4), 'dg3')
db3 = bias_ifn((ndf*4), 'db3'); dw4 = difn((ndf*8, ndf*4, 5, 5), 'dw4')
dg4 = gain_ifn((ndf*8), 'dg4'); db4 = bias_ifn((ndf*8), 'db4'); dwy = difn((ndf*8*4*4, ny), 'dwy'); 
discrim_params = [dw, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dwy]


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
    
X = T.tensor4(); Y = T.matrix()

Yhat = discrim(X, *discrim_params)

cost = bce(Yhat,Y).mean()

lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
d_updates = d_updater(discrim_params, cost)

print 'COMPILING'
t = time()
_train_d = theano.function([X, Y], cost, updates=d_updates, on_unused_input='ignore')
print '%.2f seconds to compile theano functions'%(time()-t)



print desc.upper()
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()
for epoch in range(niter):
    vaX, vaY = shuffle(vaX, vaY)
    for idx in tqdm(xrange(0, ntrain/nbatch), total=ntrain/nbatch):
        imb = vaX[idx*nbatch:((idx+1)*nbatch)]
        ymb = floatX(vaY[idx*nbatch:((idx+1)*nbatch)])
         
        a_cost = _train_d(imb, ymb)
        n_updates += 1
        n_examples += len(imb)
    
    print "Epoch =", epoch
    print "Cost vec =", np.asarray(a_cost)

        
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
        
    joblib.dump([p.get_value() for p in discrim_params], 'models/%s/discrim_params.jl'%(desc_full))
    
def load_SSGAN(desc_full):
    discrim_params = [sharedX(p) for p in joblib.load('models/%s/discrim_params.jl'%(desc_full))]
    X = T.tensor4(); 
    dX = discrim(X, *discrim_params)
    return dX
