#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:44:20 2017

@author: skumar
"""


import sys
sys.path.append('..')

import argparse
import numpy as np
from sklearn.externals import joblib

from numpy.random import RandomState
from random import Random

import theano
import theano.tensor as T

from lib import activations
from lib.ops import batchnorm, deconv
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot

from keras.datasets import mnist

    
parser = argparse.ArgumentParser()
parser.add_argument('--seed_list', default=[1])
parser.add_argument('--seed_data_list', default=[1])
parser.add_argument('--ss_count', default=20)
parser.add_argument('--s_count', default=50000)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--labeled_weight', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--niter', default=3)
parser.add_argument('--batch_loops', default=10)
parser.add_argument('--adam_recon_loops', default=500)
args = parser.parse_args()
    
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
nlab = args.s_count # # number of labeled examples
alpha = args.labeled_weight

py_rng = Random(args.seed_list[0])
np_rng = RandomState(args.seed_list[0])
rng_data = RandomState(args.seed_data_list[0])

taX, taY = mnist.load_data()[1]
taX = floatX(taX)[:,:,:,np.newaxis]
taX = taX.transpose(0, 3, 1, 2)/255.0 - 0.
taY = np.array(taY)
ntest = len(taY)
ntest = ntest/nbatch * nbatch
taX = taX[:ntest]; taY = taY[:ntest]
inds = rng_data.permutation(ntest)
taX = taX[inds]; taY = taY[inds]
taY = floatX(OneHot(taY,ny))

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy
cce = T.nnet.categorical_crossentropy
softmax = activations.Softmax()

def evaluate_reconstruction_error(args):

    def gen(Z, Y, w, w2, w3, wx):
        Z = T.concatenate([Z, Y], axis=1)
        h = relu(batchnorm(T.dot(Z, w)))
        h2 = relu(batchnorm(T.dot(h, w2)))
        h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))
        h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
        x = sigmoid(deconv(h3, wx, subsample=(2, 2), border_mode=(2, 2)))
        return x
    
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
        for i in range(300):
            cost = optim()
            min_cost = min(cost,min_cost)
            #print i, cost
        
        return cost, min_cost
    
    ss_s_score = ss_c_score = ss_ac_score = c_score = ac_score = 0.

    for sd in args.seed_data_list:
		for s in args.seed_list:
            
			args.seed_data = sd
			args.seed = s

			ss_c_desc = 'mnist_SS_Cgan_'+str(args.ss_count)+'_'+'alp'+'_'+str(args.seed)+'_'+str(args.seed_data)
			ss_ac_desc = 'mnist_SS_ACgan_'+str(args.ss_count)+'_'+str(alpha)+'_'+str(args.seed)+'_'+str(args.seed_data)
			ss_s_desc = 'mnist_SS_Sgan_'+str(args.ss_count)+'_'+str(alpha)+'_'+str(args.seed)+'_'+str(args.seed_data)
	
			ss_c_gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(ss_c_desc))]
			ss_ac_gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(ss_ac_desc))]
			ss_s_gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(ss_s_desc))]
			
			for idx in range(args.batch_loops):
				sample_zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, nz)))
				sample_ymb = floatX(taY[idx*nbatch:((idx+1)*nbatch)])

				true_samples = taX[idx*nbatch:((idx+1)*nbatch)]
				sY = theano.shared(sample_ymb)
				sZ = theano.shared(sample_zmb)                 
			
				ss_s_score += optimize(sZ, sY, ss_s_gen_params, true_samples)[1]
				ss_c_score += optimize(sZ, sY, ss_c_gen_params, true_samples)[1]
				ss_ac_score += optimize(sZ, sY, ss_ac_gen_params, true_samples)[1]


    
    args.seed_data = args.seed_data_list[0]
    args.seed = args.seed_list[0]

    c_desc = 'mnist_Cgan_'+str(args.s_count)+'_'+'alp'+'_'+str(args.seed)+'_'+str(args.seed_data)
    ac_desc = 'mnist_ACgan_'+str(args.s_count)+'_'+str(alpha)+'_'+str(args.seed)+'_'+str(args.seed_data)

    c_gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(c_desc))]
    ac_gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(ac_desc))]
    

    for idx in range(args.batch_loops):
        sample_zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, nz)))
        sample_ymb = floatX(taY[idx*nbatch:((idx+1)*nbatch)])

        true_samples = taX[idx*nbatch:((idx+1)*nbatch)]
        sY = theano.shared(sample_ymb)
        sZ = theano.shared(sample_zmb)                 
    
        c_score += optimize(sZ, sY, c_gen_params, true_samples)[1]
        ac_score += optimize(sZ, sY, ac_gen_params, true_samples)[1]

    ss_s_score /= len(args.seed_data_list)*len(args.seed_list)*args.batch_loops
    ss_c_score /= len(args.seed_data_list)*len(args.seed_list)*args.batch_loops
    ss_ac_score /= len(args.seed_data_list)*len(args.seed_list)*args.batch_loops
    c_score /= args.batch_loops
    ac_score /= args.batch_loops


    return ss_s_score, ss_c_score, ss_ac_score, c_score, ac_score

if __name__=='__main__':
	#train_models(args)
	ss_s_score, ss_c_score, ss_ac_score, c_score, ac_score = evaluate_reconstruction_error(args)


