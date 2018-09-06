#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:13:33 2017

@author: skumar
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 13:29:41 2017

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
nlab = 51200      # # number of labeled examples

nlab = nlab/nbatch*nbatch
vaX, vaY = faces()
vaX = transform(vaX)
ntrain = len(vaX)
vaX = vaX[:nlab]; vaY = vaY[:nlab]
nfac = ntrain/nlab
niter = niter*nfac
ntrain = nlab


desc = 'C_GAN'
desc_full = desc
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

gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)

gw  = gifn((nz+1*ny, ngf*8*4*4), 'gw'); gg = gain_ifn((ngf*8*4*4), 'gg')
gb = bias_ifn((ngf*8*4*4), 'gb'); gw2 = gifn((ngf*8, ngf*4, 5, 5), 'gw2'); gg2 = gain_ifn((ngf*4), 'gg2')
gb2 = bias_ifn((ngf*4), 'gb2'); gw3 = gifn((ngf*4, ngf*2, 5, 5), 'gw3'); gg3 = gain_ifn((ngf*2), 'gg3')
gb3 = bias_ifn((ngf*2), 'gb3'); gw4 = gifn((ngf*2, ngf, 5, 5), 'gw4'); gg4 = gain_ifn((ngf), 'gg4')
gb4 = bias_ifn((ngf), 'gb4'); gwx = gifn((ngf, nc, 5, 5), 'gwx')

dw  = difn((ndf, nc, 5, 5), 'dw'); dw2 = difn((ndf*2, ndf+1*ny, 5, 5), 'dw2'); dg2 = gain_ifn((ndf*2), 'dg2')
db2 = bias_ifn((ndf*2), 'db2'); dw3 = difn((ndf*4, ndf*2, 5, 5), 'dw3'); dg3 = gain_ifn((ndf*4), 'dg3')
db3 = bias_ifn((ndf*4), 'db3'); dw4 = difn((ndf*8, ndf*4, 5, 5), 'dw4')
dg4 = gain_ifn((ndf*8), 'dg4'); db4 = bias_ifn((ndf*8), 'db4'); dwy = difn((ndf*8*4*4, 1), 'dwy'); dwq = difn((ndf*8*4*4, 1), 'dwq')

gen_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gw4, gg4, gb4, gwx]
discrim_params = [dw, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dwy]

def gen(Z, Y, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wx):
    Z = T.concatenate([Z, Y], axis=1)
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf*8, 4, 4))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = relu(batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    x = tanh(deconv(h4, wx, subsample=(2, 2), border_mode=(2, 2)))      
    return x

def discrim(X, Y, w, w2, g2, b2, w3, g3, b3, w4, g4, b4, wy):
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    h = conv_cond_concat(h, yb)
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))        
    h4 = T.flatten(h4, 2)
    p = sigmoid(T.dot(h4,wy)) # p = prob. of image being real or fake
    return p
    
W = T.tensor4(); X = T.tensor4(); Z = T.matrix(); Y = T.matrix(); Ys = T.matrix()

gX = gen(Z, Y, *gen_params)

p_real = discrim(X, Y, *discrim_params)
p_gen = discrim(gX, Y, *discrim_params)
p_match = discrim(X, Ys, *discrim_params)

d_cost_real = bce(p_real, T.ones(p_real.shape)).mean() 
d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
d_cost_match = bce(p_match, T.zeros(p_match.shape)).mean() 

g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean() 

d_cost = d_cost_real + d_cost_gen + d_cost_match
g_cost = g_cost_d

cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen, d_cost_match]
cost += [p_real.mean(), p_gen.mean(), p_match.mean()]

lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates

print 'COMPILING'
t = time()
_train_g = theano.function([X, Z, Y, Ys], cost, updates=g_updates, on_unused_input='ignore')
_train_d = theano.function([X, Z, Y, Ys], cost, updates=d_updates, on_unused_input='ignore')
_gen = theano.function([Z, Y], gX, on_unused_input='ignore')
print '%.2f seconds to compile theano functions'%(time()-t)

vis_idxs = py_rng.sample(np.arange(len(vaX)), nvis)
vaX_vis = inverse_transform(vaX[vis_idxs])
color_grid_vis(vaX_vis, (nv, nv), 'samples/%s_etl_test.png'%desc)

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(nvis, nz)))
sample_ymb = floatX(vaY[:nvis])

sample_zmb_0 = floatX(np.random.rand(ny,nz)); sample_zmb_0 = np.tile(sample_zmb_0,[ny,1])
sample_ymb_0 = floatX(OneHot(np.asarray([[i for _ in range(ny)] for i in range(ny)]).flatten(), ny))


print desc.upper()
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()
for epoch in range(niter):
    vaX, vaY = shuffle(vaX, vaY); vaYshuffle = shuffle(vaY); laX, laY = shuffle(vaX, vaY); 
    for idx in tqdm(xrange(0, ntrain/nbatch), total=ntrain/nbatch):
        imb = vaX[idx*nbatch:(idx+1)*nbatch]
        ymb = floatX(vaY[idx*nbatch:(idx+1)*nbatch])
        ymbshuffle = floatX(vaYshuffle[idx*nbatch:(idx+1)*nbatch])
        zmb = floatX(np_rng.uniform(-1., 1., size=(len(imb), nz)))
        
        if n_updates % (k+1) == 0:
            cost = _train_g(imb, zmb, ymb, ymbshuffle)
        else:
            cost = _train_d(imb, zmb, ymb, ymbshuffle)
        n_updates += 1
        n_examples += len(imb)
    
    if n_epochs%nfac==0:
        print "Epoch =", epoch/nfac
        print "Cost vec =", np.asarray(cost)[:5]
        print "pq vec =", np.asarray(cost)[5:]

        samples = np.asarray(_gen(sample_zmb, sample_ymb))
        samples0 = np.asarray(_gen(sample_zmb_0, sample_ymb_0))
    
    
        color_grid_vis(inverse_transform(samples), (nv, nv), 'samples/%s/G_%d.png'%(desc, n_epochs/nfac))
        color_grid_vis(inverse_transform(samples0), (ny, ny), 'samples/%s/G_%d_C.png'%(desc, n_epochs/nfac))
        
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
        
    joblib.dump([p.get_value() for p in gen_params], 'models/%s/gen_params.jl'%(desc_full))
    joblib.dump([p.get_value() for p in discrim_params], 'models/%s/discrim_params.jl'%(desc_full))

