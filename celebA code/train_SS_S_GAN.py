#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 09:04:29 2017

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
alpha = 1.0
offset = 00000


nlab = nlab/nbatch*nbatch
vaX, vaY = faces()
ntrain = len(vaX)
vaX = transform(vaX)

laX = vaX[:nlab]; laY = floatX(vaY[:nlab])

sample_vec = np.random.choice(range(len(laX)),len(vaX),replace=True)
laX = laX[sample_vec]; laY = laY[sample_vec]

#laX = vaX[offset:offset+nlab]; laY = vaY[offset:offset+nlab]


desc = 'SS_S_GAN'
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

dw  = difn((ndf, nc, 5, 5), 'dw'); dw2 = difn((ndf*2, ndf+0*ny, 5, 5), 'dw2'); dg2 = gain_ifn((ndf*2), 'dg2')
db2 = bias_ifn((ndf*2), 'db2'); dw3 = difn((ndf*4, ndf*2, 5, 5), 'dw3'); dg3 = gain_ifn((ndf*4), 'dg3')
db3 = bias_ifn((ndf*4), 'db3'); dw4 = difn((ndf*8, ndf*4, 5, 5), 'dw4')
dg4 = gain_ifn((ndf*8), 'dg4'); db4 = bias_ifn((ndf*8), 'db4');
dw5 = difn((ndf*8, ndf*8 + ny, 3, 3), 'dw5'); dg5 = gain_ifn((ndf*8), 'dg5'); db5 = bias_ifn((ndf*8), 'db5');
dwy = difn((ndf*8*4*4, 1), 'dwy'); dwq = difn((ndf*8*4*4, 1), 'dwq')

gen_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gw4, gg4, gb4, gwx]
margin_params = [dw, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4,dwy]
condit_params = [dw5,dg5,db5,dwq]
discrim_params = [dw, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dw5, dg5, db5, dwy, dwq]

def gen(Z, Y, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wx):
    Z = T.concatenate([Z, Y], axis=1)
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf*8, 4, 4))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = relu(batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    x = tanh(deconv(h4, wx, subsample=(2, 2), border_mode=(2, 2)))      
    return x

def discrim(X, Y, w, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5, wy, wq):
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    #h = conv_cond_concat(h, yb)
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))        
    h4f = T.flatten(h4, 2)
    p = sigmoid(T.dot(h4f,wy)) # p = prob. of image being real or fake
    
    h4 = conv_cond_concat(h4, yb)
    h4 = lrelu(batchnorm(dnn_conv(h4, w5, subsample=(1, 1), border_mode=(1, 1)), g=g5, b=b5)) 
    h4f = T.flatten(h4, 2)
    q = sigmoid(T.dot(h4f,wq)) # q = prob. of attribute given image being real or fake
    return p, q
    
W = T.tensor4(); X = T.tensor4(); Z = T.matrix(); Y = T.matrix(); Ys = T.matrix(); a=T.scalar()

gX = gen(Z, Y, *gen_params)

p_real_u, q_real_u = discrim(W, Ys*(1-Y), *discrim_params); pq_real_u = p_real_u*q_real_u
p_real_l, q_real_l = discrim(X, Y, *discrim_params); pq_real_l = p_real_l*q_real_l
p_gen, q_gen = discrim(gX, Y, *discrim_params); pq_gen = p_gen*q_gen
p_match, q_match = discrim(X, Ys*(1-Y), *discrim_params); pq_match = p_match*q_match
#p_match_u, q_match_u = discrim(W, Ys, *discrim_params)
p_gen_s, q_gen_s = discrim(gX, Ys*(1-Y), *discrim_params);
#make p_real and p_match the same?
d_cost_real_u = bce(p_real_u, 1.0*T.ones(p_real_u.shape)).mean()# + bce(p_real_l, 1.0*T.ones(p_real_l.shape)).mean())/2.
d_cost_real_l = bce(q_real_l, 1.0*T.ones(q_real_l.shape)).mean()
d_cost_gen_u = bce(p_gen, T.zeros(p_gen.shape)).mean() 
d_cost_gen_l = bce(q_gen, T.zeros(q_gen.shape)).mean()
d_cost_match = bce(q_match, T.zeros(q_match.shape)).mean() 
d_cost_eq = T.mean(abs(p_real_l-p_match)) #+ T.mean(abs(p_real_u-p_match_u))
 
d_cost_s = d_cost_real_l + d_cost_gen_l #+ d_cost_match)/2. #+ a#((1-a)*d_cost_match + 1*d_cost_gen_l)/2.
d_cost_u = d_cost_real_u + d_cost_gen_u
g_cost_u = bce(p_gen, 1.0*T.ones(p_gen.shape)).mean() 
g_cost_s = bce(q_gen, 1.0*T.ones(q_gen.shape)).mean() #+ bce(q_gen_s, T.zeros(q_gen_s.shape)).mean()

d_cost = 1*d_cost_u + 1*alpha*d_cost_s #+ d_cost_eq
g_cost = 1*g_cost_u + 1*a*alpha*g_cost_s + 100*T.maximum(0,abs(T.mean(abs(W))-T.mean(abs(gX)))-0.1)


cost = [g_cost, d_cost, g_cost_s, g_cost_u, d_cost_real_u, d_cost_real_l, d_cost_gen_u, d_cost_match]
cost += [p_real_u.mean(), q_real_u.mean(), p_real_l.mean(), q_real_l.mean(), p_gen.mean(), q_gen.mean()]

lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
d_updates_u = d_updater(margin_params, d_cost_u)
g_updates_u = g_updater(gen_params, g_cost_u)
d_updates_s = d_updater(condit_params, d_cost_s)
g_updates_s = g_updater(gen_params, g_cost)
#updates = d_updates + g_updates

print 'COMPILING'
t = time()
_train_g_u = theano.function([X, Z, Y, Ys, W, a], cost, updates=g_updates_u, on_unused_input='ignore')
_train_d_u = theano.function([X, Z, Y, Ys, W, a], cost, updates=d_updates_u, on_unused_input='ignore')
_train_g_s = theano.function([X, Z, Y, Ys, W, a], cost, updates=g_updates_s, on_unused_input='ignore')
_train_d_s = theano.function([X, Z, Y, Ys, W, a], cost, updates=d_updates_s, on_unused_input='ignore')
_gen = theano.function([Z, Y], gX, on_unused_input='ignore')
print '%.2f seconds to compile theano functions'%(time()-t)

vis_idxs = py_rng.sample(np.arange(len(vaX)), nvis)
vaX_vis = inverse_transform(vaX[vis_idxs])
color_grid_vis(vaX_vis, (nv, nv), 'samples/%s_etl_test.png'%desc)

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(np.minimum(nvis,len(vaY)), nz)))
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
    vaX, vaY = shuffle(vaX, vaY); vaYshuffle = shuffle(vaY); laX, laY = shuffle(laX, laY); 
    for idx in tqdm(xrange(0, ntrain/nbatch), total=ntrain/nbatch):
        wmb = vaX[idx*nbatch:(idx+1)*nbatch]
        #if (idx+1)*nbatch%nlab==0:
        #    laX, laY = shuffle(laX, laY); 
        imb = laX[idx*nbatch:(idx+1)*nbatch]
        ymb = floatX(laY[idx*nbatch:(idx+1)*nbatch])
        ymbshuffle = floatX(vaYshuffle[idx*nbatch:(idx+1)*nbatch])
        zmb = floatX(np_rng.uniform(-1., 1., size=(len(imb), nz)))
        
        aval = 0.15 #min(2*float(epoch)/niter,1)        
        if n_updates % (k+1) == 0:
            #cost = _train_g_u(imb, zmb, ymb, ymbshuffle, wmb, aval)
            cost = _train_g_s(imb, zmb, ymb, ymbshuffle, wmb, aval)
        else:
            cost = _train_d_u(imb, zmb, ymb, ymbshuffle, wmb, aval)
            cost = _train_d_s(imb, zmb, ymb, ymbshuffle, wmb, aval)
        n_updates += 1
        n_examples += len(imb)
    
    print "Epoch =", epoch
    print "Cost vec =", np.asarray(cost)[:8]
    print "pq vec =", np.asarray(cost)[8:]

    samples = np.asarray(_gen(sample_zmb, sample_ymb))
    samples0 = np.asarray(_gen(sample_zmb_0, sample_ymb_0))
        
    color_grid_vis(inverse_transform(samples), (nv, nv), 'samples/%s/G_%d.png'%(desc, n_epochs))
    color_grid_vis(inverse_transform(samples0), (ny, ny), 'samples/%s/G_%d_C.png'%(desc, n_epochs))
        
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
        
    joblib.dump([p.get_value() for p in gen_params], 'models/%s/gen_params.jl'%(desc_full))
    joblib.dump([p.get_value() for p in discrim_params], 'models/%s/discrim_params.jl'%(desc_full))
    
