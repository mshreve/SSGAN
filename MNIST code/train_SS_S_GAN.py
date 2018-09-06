#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:45:32 2017

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


# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--seed_data', default=1)
parser.add_argument('--count', default=20)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--labeled_weight', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--niter', default=150)
args = parser.parse_args()

def SS_S_GAN(args):
    
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
    alpha = args.labeled_weight
    
    py_rng = Random(args.seed)
    np_rng = RandomState(args.seed)
    rng_data = RandomState(args.seed_data)

    vaX, vaY = mnist.load_data()[0]
    vaX = floatX(vaX)[:,:,:,np.newaxis]
    vaX = vaX.transpose(0, 3, 1, 2)/255.0 - 0.
    vaY = np.array(vaY)
    ntrain = len(vaX)
    ntrain = ntrain/nbatch * nbatch
    vaX = vaX[:ntrain]; vaY = vaY[:ntrain]
    inds = rng_data.permutation(ntrain)
    vaX = vaX[inds]; vaY = vaY[inds]
    
    def transform(X):
        X = [center_crop(x, npx) for x in X]
        return floatX(X).transpose(0, 3, 1, 2)/255.0 - 0.

    def inverse_transform(X):
        X = (X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1)+0.)/1.
        return X

    ndig = nlab/ny
    vlist = []
    for i in range(ny):
        vlist += list(np.where(vaY==i)[0][:ndig])

    vaY = floatX(OneHot(vaY,ny))
    laX = vaX[vlist]; laY=vaY[vlist]
    sample_vec = np.random.choice(range(len(laX)),len(vaX),replace=True)
    laX = laX[sample_vec]; laY = laY[sample_vec]
    nlab = len(vlist)


    #desc = 'mnist_semisup_DC_dcgan'
    desc = 'mnist_SS_Sgan_'+str(args.count)+'_'+str(alpha)+'_'+str(args.seed)+'_'+str(args.seed_data)
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

    dw  = difn((ndf, nc, 5, 5), 'dw'); dw2 = difn((ndf*2, ndf, 5, 5), 'dw2')
    dw3 = difn((ndf*2*7*7, ndfc), 'dw3'); dwy = difn((ndfc, 1), 'dwy')

    cw  = difn((ndf, nc+ny, 5, 5), 'cw'); cw2 = difn((ndf*2, ndf+0*ny, 5, 5), 'cw2')
    cw3 = difn((ndf*2*7*7+0*ny, ndfc), 'cw3'); cwy = difn((ndfc+0*ny, 1), 'cwy')

    cw  = difn((ndfc+1*ny, ndfc/2), 'cw'); cw2  = difn((ndfc/2, ndfc/4), 'cw2');
    cw3  = difn((ndfc/4, ndfc/8), 'cw3'); cwy = difn((ndfc/8, 1), 'cwy')


    gen_params = [gw, gw2, gw3, gwx]
    margin_params = [dw, dw2, dw3, dwy]
    condit_params = [cw, cw2, cw3, cwy]
    discrim_params = margin_params+condit_params

    def gen(Z, Y, w, w2, w3, wx):
        Z = T.concatenate([Z, Y], axis=1)
        h = relu(batchnorm(T.dot(Z, w)))
        h2 = relu(batchnorm(T.dot(h, w2)))
        h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))
        h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
        x = sigmoid(deconv(h3, wx, subsample=(2, 2), border_mode=(2, 2)))
        return x

    def discrim(X, Y, w, w2, w3, wy, cw, cw2, cw3, cwy):
        h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
        h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))))
        h2 = T.flatten(h2, 2)
        h3 = lrelu(batchnorm(T.dot(h2, w3)))
        p = sigmoid(T.dot(h3, wy))
        
        h3 = T.concatenate([h3,Y], axis=1)
        h4 = lrelu(T.dot(h3, cw))
        h5 = lrelu(T.dot(h4, cw2))
        h6 = lrelu(T.dot(h5, cw3))
        q = sigmoid(T.dot(h6, cwy))
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
     
    d_cost_s = d_cost_real_l + d_cost_gen_l #+ a#((1-a)*d_cost_match + 1*d_cost_gen_l)/2.
    d_cost_u = d_cost_real_u + d_cost_gen_u
    g_cost_u = bce(p_gen, 1.0*T.ones(p_gen.shape)).mean() 
    g_cost_s = bce(q_gen, 1.0*T.ones(q_gen.shape)).mean() #+ bce(q_gen_s, T.zeros(q_gen_s.shape)).mean()

    d_cost = 1*d_cost_u + 1*alpha*d_cost_s #+ d_cost_eq
    g_cost = 1*g_cost_u + 1*a*alpha*g_cost_s #+ 100*T.maximum(0,abs(T.mean(abs(W))-T.mean(abs(gX)))-0.1)


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

    vis_idxs = py_rng.sample(np.arange(len(laX)), nvis)
    vaX_vis = inverse_transform(laX[vis_idxs])
    color_grid_vis(vaX_vis, (nv, nv), 'samples/%s_etl_test.png'%desc)

    sample_zmb = floatX(np_rng.uniform(-1., 1., size=(np.minimum(nvis,len(vaY)), nz)))
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
        vaX, vaY = shuffle(vaX, vaY); vaYshuffle = shuffle(vaY); laX, laY = shuffle(laX, laY); 
        num_a = floatX(0.); den_a = floatX(0.); aval = floatX(1.)
        for idx in tqdm(xrange(0, ntrain/nbatch), total=ntrain/nbatch):
            wmb = vaX[idx*nbatch:(idx+1)*nbatch]
            #if (idx+1)*nbatch%nlab==0:
            #    laX, laY = shuffle(laX, laY); 
            imb = laX[idx*nbatch:(idx+1)*nbatch]
            ymb = floatX(laY[idx*nbatch:(idx+1)*nbatch])
            ymbshuffle = floatX(vaYshuffle[idx*nbatch:(idx+1)*nbatch])
            zmb = floatX(np_rng.uniform(-1., 1., size=(len(imb), nz)))
            
            #aval = 0 #min(2*float(epoch)/niter,1)        
            if n_updates % (k+1) == 0:
                #cost = _train_g_u(imb, zmb, ymb, ymbshuffle, wmb, aval)
                cost = _train_g_s(imb, zmb, ymb, ymbshuffle, wmb, aval)
                num_a = cost[3]; den_a = cost[2] + eps; aval = 0.15; #floatX(num_a/den_a); 
            else:
                cost = _train_d_u(imb, zmb, ymb, ymbshuffle, wmb, aval)
                cost = _train_d_s(imb, zmb, ymb, ymbshuffle, wmb, aval)
            n_updates += 1
            n_examples += len(imb)
        
        print "Epoch =", epoch
        print "Cost vec =", np.asarray(cost)[:8]
        print "pq vec =", np.asarray(cost)[8:]
        print aval

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
    SS_S_GAN(args)