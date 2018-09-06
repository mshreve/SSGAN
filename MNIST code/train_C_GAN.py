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

import argparse
import os
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

from numpy.random import RandomState
from random import Random

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from theano.tensor.extra_ops import diff


from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch

from keras.datasets import mnist

    
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--seed_data', default=1)
parser.add_argument('--count', default=50000)
parser.add_argument('--batch_size', default=128)
#parser.add_argument('--labeled_weight', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--niter', default=150)
args = parser.parse_args()

def C_GAN(args):
    
    k = 1             # # of discrim updates for each gen update
    l2 = 2.5e-5       # l2 weight decay
    nv = 30;nvis = 900# # of samples to visualize during training
    b1 = 0.5          # momentum term of adam
    nc = 1            # # of channels in image
    nbatch = args.batch_size # # of examples in batch
    npx = 28          # # of pixels width/height of images
    nz = 25           # # of dim for Z
    ny = 10           # # of dim for Y
    ngf = 64          # # of gen filters in first conv layer
    ndf = 64          # # of discrim filters in first conv layer
    ngfc = 1024       # # of gen units for fully connected layers
    ndfc = 1024       # # of discrim units for fully connected layers
    nx = npx*npx*nc   # # of dimensions in X
    niter = args.niter       # # of iter at starting learning rate
    niter_decay = 0   # # of iter to linearly decay learning rate to zero
    lr = args.learning_rate # initial learning rate for adam
    eps = 10**-20     # # to avoid division of zero errors
    nlab = args.count # # number of labeled examples
    #alpha = args.labeled_weight
    py_rng = Random(args.seed)
    np_rng = RandomState(args.seed)
    rng_data = RandomState(args.seed_data)

    vaX, vaY = mnist.load_data()[0]
    vaX = floatX(vaX)[:,:,:,np.newaxis]
    vaX = vaX.transpose(0, 3, 1, 2)/255.0 - 0.
    ntrain = min(len(vaX),nlab)
    vaY = np.array(vaY)
    ntrain = ntrain/nbatch * nbatch
    vaX = vaX[:ntrain]; vaY = vaY[:ntrain]
    inds = rng_data.permutation(ntrain)
    vaX = vaX[inds]; vaY = vaY[inds]
    vaY = floatX(OneHot(vaY,ny))
    
    def transform(X):
        X = [center_crop(x, npx) for x in X]
        return floatX(X).transpose(0, 3, 1, 2)/255.0 - 0.

    def inverse_transform(X):
        X = (X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1)+0.)/1.
        return X

    #desc = 'mnist_smallsup_dcgan'
    desc = 'mnist_Cgan_'+str(args.count)+'_'+'alp'+'_'+str(args.seed)+'_'+str(args.seed_data)
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
    softmax = activations.Softmax()
    lrelu = activations.LeakyRectify()
    tanh = activations.Tanh()
    bce = T.nnet.binary_crossentropy
    cce = T.nnet.categorical_crossentropy

    gifn = inits.Normal(scale=0.02)
    difn = inits.Normal(scale=0.02)

    gw  = gifn((nz+1*ny, ngfc), 'gw'); gw2 = gifn((ngfc, ngf*2*7*7), 'gw2'); 
    gw3 = gifn((ngf*2, ngf, 5, 5), 'gw3'); gwx = gifn((ngf, 1, 5, 5), 'gwx')

    dw  = difn((ndf, nc+ny, 5, 5), 'dw'); dw2 = difn((ndf*2, ndf, 5, 5), 'dw2')
    dw3 = difn((ndf*2*7*7, ndfc), 'dw3'); dwy = difn((ndfc, 1), 'dwy')

    gen_params = [gw, gw2, gw3, gwx]
    discrim_params = [dw, dw2, dw3, dwy]

    def gen(Z, Y, w, w2, w3, wx):
        Z = T.concatenate([Z, Y], axis=1)
        h = relu(batchnorm(T.dot(Z, w)))
        h2 = relu(batchnorm(T.dot(h, w2)))
        h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))
        h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
        x = sigmoid(deconv(h3, wx, subsample=(2, 2), border_mode=(2, 2)))
        return x

    def discrim(X, Y, w, w2, w3, wy):    
        yb = Y.dimshuffle(0, 1, 'x', 'x')
        X = conv_cond_concat(X, yb)
        h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
        #h = conv_cond_concat(h, yb)
        h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))))
        h2 = T.flatten(h2, 2)
        #h2 = T.concatenate([h2, Y], axis=1)
        h3 = lrelu(batchnorm(T.dot(h2, w3)))
        #h3 = T.concatenate([h3, Y], axis=1)
        q = sigmoid(T.dot(h3, wy))
        return q
        
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
    #updates = d_updates + g_updates

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

    sample_zmb_0 = floatX(np.random.rand(ny,nz)); sample_zmb_0 = np.tile(sample_zmb_0,[nv,1])
    sample_ymb_0 = floatX(OneHot(np.asarray([[i for _ in range(nv)] for i in range(ny)]).flatten(), ny))


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
        
        print "Epoch =", epoch
        print "Cost vec =", np.asarray(cost)[:5]
        print "pq vec =", np.asarray(cost)[5:]

        samples = np.asarray(_gen(sample_zmb, sample_ymb))
        samples0 = np.asarray(_gen(sample_zmb_0, sample_ymb_0))
        
        
        color_grid_vis(inverse_transform(samples), (nv, nv), 'samples/%s/G_%d.png'%(desc, n_epochs))
        color_grid_vis(inverse_transform(samples0), (ny, nv), 'samples/%s/G_%d_C.png'%(desc, n_epochs))
            
        n_epochs += 1
        if n_epochs > niter:
            lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
            
        joblib.dump([p.get_value() for p in gen_params], 'models/%s/gen_params.jl'%(desc))
        joblib.dump([p.get_value() for p in discrim_params], 'models/%s/discrim_params.jl'%(desc))

if __name__ =='__main__':
    C_GAN(args)