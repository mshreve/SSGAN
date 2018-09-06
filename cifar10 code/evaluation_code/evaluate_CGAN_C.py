#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:52:55 2017

@author: skumar

code for evaluating GAN's - identify closest samples in model. 
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

desc = 'cifar_attr_predict'
desc_full = 'cifar_attr_predict'
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
softmax = activations.Softmax()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy
cce = T.nnet.categorical_crossentropy


X = T.tensor4(); Z = T.matrix(); Y = T.matrix()

def gen(Z, Y, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wx):
    Z = T.concatenate([Z, Y], axis=1)
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf*8*2*1, 2, 2))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = relu(batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    x = tanh(deconv(h4, wx, subsample=(2, 2), border_mode=(2, 2)))      
    return x

desc_full = 'SS_AC_GAN'
ac_gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(desc_full))]
ac_gX = gen(Z, Y, *ac_gen_params)
ac_gen = theano.function([Z, Y], ac_gX, on_unused_input='ignore')

desc_full = 'AC_GAN'
am_gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(desc_full))]
am_gX = gen(Z, Y, *am_gen_params)
am_gen = theano.function([Z, Y], am_gX, on_unused_input='ignore')

desc_full = 'SS_S_GAN'
ss_gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(desc_full))]
ss_gX = gen(Z, Y, *ss_gen_params)
ss_gen = theano.function([Z, Y], ss_gX, on_unused_input='ignore')

desc_full = 'C_GAN'
s_gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(desc_full))]
s_gX = gen(Z, Y, *s_gen_params)
s_gen = theano.function([Z, Y], s_gX, on_unused_input='ignore')

desc_full = 'SS_C_GAN'
sm_gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(desc_full))]
sm_gX = gen(Z, Y, *sm_gen_params)
sm_gen = theano.function([Z, Y], sm_gX, on_unused_input='ignore')


def Adam(cost, params, lr=0.02, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(floatX(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates


#desc_full = 'sup_dcgan_1000'
#gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(desc_full))]
#$s_gX = gen(Z, Y, *gen_params)


def optimize(Z, Y, gen_params, true_samp):
    #yb = Y.dimshuffle(0, 1, 'x', 'x')
    tZ = tanh(Z)
    tY = Y#T.switch(Y>0,1.,0.)
    syn_samp = gen(tZ,tY,*gen_params)
    cost = T.mean((syn_samp-true_samp)**2)
    #gZ, gY = T.grad(cost,[Z])
    
    updates = Adam(cost,[Z])
    optim = theano.function([],outputs=cost, updates=updates)#((Z, Z-0.1*gZ),(Y, Y-0.1*gY)))
    
    min_cost = 1
    for i in range(2500):
        cost = optim()
        min_cost = min(cost,min_cost)
        print i, cost
        
    #print cost
    
    return cost, min_cost, Z
        

nvis = 128
offset = 13000

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(nvis, nz)))
sample_ymb = floatX(vaY[offset:offset+nvis])

true_samples = vaX[offset:offset+nvis]
sZ = theano.shared(sample_zmb)
sY = theano.shared(sample_ymb)


am_cost = optimize(sZ, sY, am_gen_params, true_samples)
ac_cost = optimize(sZ, sY, ac_gen_params, true_samples)
ss_cost = optimize(sZ, sY, ss_gen_params, true_samples)
sm_cost = optimize(sZ, sY, sm_gen_params, true_samples)
s_cost = optimize(sZ, sY, s_gen_params, true_samples)

am_sample_zmb = np.tanh(am_cost[2].get_value())
ac_sample_zmb = np.tanh(ac_cost[2].get_value())
ss_sample_zmb = np.tanh(ss_cost[2].get_value())
sm_sample_zmb = np.tanh(sm_cost[2].get_value())
s_sample_zmb = np.tanh(s_cost[2].get_value())


ac_samples = np.asarray(ac_gen(ac_sample_zmb, sample_ymb))
am_samples = np.asarray(am_gen(am_sample_zmb, sample_ymb))
ss_samples = np.asarray(ss_gen(ac_sample_zmb, sample_ymb))
s_samples_small = np.asarray(sm_gen(sm_sample_zmb, sample_ymb))
s_samples = np.asarray(s_gen(s_sample_zmb, sample_ymb))

from skimage.measure import compare_ssim as ssim
ac_score = am_score = ss_score = sm_score = s_score = 0.
for i in range(128):
    ts = true_samples[0].transpose(1,2,0)
    acs = ac_samples[0].transpose(1,2,0)
    ams = am_samples[0].transpose(1,2,0)
    sss = ss_samples[0].transpose(1,2,0)
    sms = s_samples_small[0].transpose(1,2,0)
    ss = s_samples[0].transpose(1,2,0)
    
    ac_score += ssim(ts, acs, multichannel=True, data_range=2)
    am_score += ssim(ts, ams, multichannel=True, data_range=2)
    ss_score += ssim(ts, sss, multichannel=True, data_range=2)
    sm_score += ssim(ts, sms, multichannel=True, data_range=2)
    s_score += ssim(ts, ss, multichannel=True, data_range=2)
    
ac_score /= 128
am_score /= 128
ss_score /= 128
sm_score /= 128
s_score /= 128

print ac_score, am_score, ss_score, sm_score, s_score
print ac_cost[1], am_cost[1], ss_cost[1], sm_cost[1], s_cost[1]




