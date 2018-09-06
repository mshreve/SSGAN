#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:27:51 2017

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
from theano.sandbox.cuda.dnn import dnn_conv

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

def evaluate_attribute_prediction(args):

    X = T.tensor4(); Z = T.matrix(); Y = T.matrix()

    def gen(Z, Y, w, w2, w3, wx):
        Z = T.concatenate([Z, Y], axis=1)
        h = relu(batchnorm(T.dot(Z, w)))
        h2 = relu(batchnorm(T.dot(h, w2)))
        h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))
        h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
        x = sigmoid(deconv(h3, wx, subsample=(2, 2), border_mode=(2, 2)))
        return x

    def discrim(X, w, w2, w3, wy):    
        h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
        h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))))
        h2 = T.flatten(h2, 2)
        h3 = lrelu(batchnorm(T.dot(h2, w3)))
        q = softmax(T.dot(h3, wy))
        return q
    
    true_score = ss_s_score = ss_c_score = ss_ac_score = c_score = ac_score = 0.

    for sd in args.seed_data_list:
		for s in args.seed_list:
            
			args.seed_data = sd
			args.seed = s

			ss_c_desc = 'mnist_SS_Cgan_'+str(args.ss_count)+'_'+'alp'+'_'+str(args.seed)+'_'+str(args.seed_data)
			ss_ac_desc = 'mnist_SS_ACgan_'+str(args.ss_count)+'_'+str(alpha)+'_'+str(args.seed)+'_'+str(args.seed_data)
			ss_s_desc = 'mnist_SS_Sgan_'+str(args.ss_count)+'_'+str(alpha)+'_'+str(args.seed)+'_'+str(args.seed_data)
	
			attr_desc = 'mnist_attr_predict_'+str(args.s_count)+'_'+'alp'+'_'+str(args.seed_list[0])+'_'+str(args.seed_data_list[0])

			gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(ss_c_desc))]
			ss_c_gX = gen(Z, Y, *gen_params); ss_c_gen = theano.function([Z, Y], ss_c_gX, on_unused_input='ignore')

			gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(ss_ac_desc))]
			ss_ac_gX = gen(Z, Y, *gen_params); ss_ac_gen = theano.function([Z, Y], ss_ac_gX, on_unused_input='ignore')

			gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(ss_s_desc))]
			ss_s_gX = gen(Z, Y, *gen_params); ss_s_gen = theano.function([Z, Y], ss_s_gX, on_unused_input='ignore')

			discrim_params = [sharedX(p) for p in joblib.load('models/%s/discrim_params.jl'%(attr_desc))]
			dX = discrim(X, *discrim_params)
			dx = theano.function([X], dX, on_unused_input='ignore')

			for idx in range(args.batch_loops):
				sample_zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, nz)))
				sample_ymb = floatX(taY[idx*nbatch:((idx+1)*nbatch)])

				ss_s_samples = np.asarray(ss_s_gen(sample_zmb, sample_ymb))
				ss_c_samples = np.asarray(ss_c_gen(sample_zmb, sample_ymb))
				ss_ac_samples = np.asarray(ss_ac_gen(sample_zmb, sample_ymb))

				ss_s_pred = dx(ss_s_samples)
				ss_c_pred = dx(ss_c_samples)
				ss_ac_pred = dx(ss_ac_samples)

				ss_s_score += np.sqrt(np.mean((ss_s_pred-sample_ymb)**2))
				ss_c_score += np.sqrt(np.mean((ss_c_pred-sample_ymb)**2))
				ss_ac_score += np.sqrt(np.mean((ss_ac_pred-sample_ymb)**2))

	        
    args.seed_data = args.seed_data_list[0]
    args.seed = args.seed_list[0]

    c_desc = 'mnist_Cgan_'+str(args.s_count)+'_'+'alp'+'_'+str(args.seed)+'_'+str(args.seed_data)
    ac_desc = 'mnist_ACgan_'+str(args.s_count)+'_'+str(alpha)+'_'+str(args.seed)+'_'+str(args.seed_data)
    attr_desc = 'mnist_attr_predict_'+str(args.s_count)+'_'+'alp'+'_'+str(args.seed)+'_'+str(args.seed_data)

    gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(c_desc))]
    c_gX = gen(Z, Y, *gen_params); c_gen = theano.function([Z, Y], c_gX, on_unused_input='ignore')

    gen_params = [sharedX(p) for p in joblib.load('models/%s/gen_params.jl'%(ac_desc))]
    ac_gX = gen(Z, Y, *gen_params); ac_gen = theano.function([Z, Y], ac_gX, on_unused_input='ignore')


    discrim_params = [sharedX(p) for p in joblib.load('models/%s/discrim_params.jl'%(attr_desc))]
    dX = discrim(X, *discrim_params)
    dx = theano.function([X], dX, on_unused_input='ignore')

    for idx in range(args.batch_loops):
        sample_zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, nz)))
        sample_ymb = floatX(taY[idx*nbatch:((idx+1)*nbatch)])

        true_samples = taX[idx*nbatch:((idx+1)*nbatch)]
        ac_samples = np.asarray(ac_gen(sample_zmb, sample_ymb))
        c_samples = np.asarray(c_gen(sample_zmb, sample_ymb))

        true_pred = dx(true_samples)
        ac_pred = dx(ac_samples)
        c_pred = dx(c_samples)

        true_score += np.sqrt(np.mean((true_pred-sample_ymb)**2))
        c_score += np.sqrt(np.mean((c_pred-sample_ymb)**2))
        ac_score += np.sqrt(np.mean((ac_pred-sample_ymb)**2))

    true_score /= args.batch_loops
    ss_s_score /= len(args.seed_data_list)*len(args.seed_list)*args.batch_loops
    ss_c_score /= len(args.seed_data_list)*len(args.seed_list)*args.batch_loops
    ss_ac_score /= len(args.seed_data_list)*len(args.seed_list)*args.batch_loops
    c_score /= args.batch_loops
    ac_score /= args.batch_loops


    return true_score, ss_s_score, ss_c_score, ss_ac_score, c_score, ac_score

if __name__=='__main__':
	true_score, ss_s_score, ss_c_score, ss_ac_score, c_score, ac_score = evaluate_attribute_prediction(args)

