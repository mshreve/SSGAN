#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 18:02:27 2017

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
from lib.ops import batchnorm, conv_cond_concat
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot
from keras.datasets import mnist


from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_predict

logreg = linear_model.LogisticRegression(C=1e5)

parser = argparse.ArgumentParser()
parser.add_argument('--seed_list', default=[1])
parser.add_argument('--seed_data_list', default=[1])
parser.add_argument('--ss_count', default=20)
parser.add_argument('--s_count', default=50000)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--discrim_batch_size', default=5000)
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
nlab = args.ss_count # # number of labeled examples
alpha = args.labeled_weight

py_rng = Random(args.seed_list[0])
np_rng = RandomState(args.seed_list[0])
rng_data = RandomState(args.seed_data_list[0])

taX, taY = mnist.load_data()[1]
taX = floatX(taX)[:,:,:,np.newaxis]
taX = taX.transpose(0, 3, 1, 2)/255.0 - 0.
taY = np.array(taY)
TAY = np.asarray(taY)
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


def evaluate_discrim_classifier(args):

    X = T.tensor4(); Y = T.matrix()

    def s_discrim(X, Y, w, w2, w3, wy):    	
	    yb = Y.dimshuffle(0, 1, 'x', 'x')
	    X = conv_cond_concat(X, yb)
	    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
	    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))))
	    h2 = T.flatten(h2, 2)
	    h3 = lrelu(batchnorm(T.dot(h2, w3)))
	    return h3

    def ac_discrim(X, Y, w, w2, w3, wy, wq):
	    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
	    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))))
	    h2 = T.flatten(h2, 2)
	    h3 = lrelu(batchnorm(T.dot(h2, w3)))
	    return h3

    def ss_discrim(X, Y, w, w2, w3, wy, cw, cw2, cw3, cwy):
	    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
	    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))))
	    h2 = T.flatten(h2, 2)
	    h3 = lrelu(batchnorm(T.dot(h2, w3)))       
	    h3 = T.concatenate([h3,Y], axis=1)
	    h4 = lrelu(T.dot(h3, cw))
	    return h4

    ss_s_score = ss_c_score = ss_ac_score = c_score = ac_score = 0.

    for sd in args.seed_data_list:
		for s in args.seed_list:
            
			args.seed_data = sd
			args.seed = s

			ss_c_desc = 'mnist_SS_Cgan_'+str(args.ss_count)+'_'+'alp'+'_'+str(args.seed)+'_'+str(args.seed_data)
			ss_ac_desc = 'mnist_SS_ACgan_'+str(args.ss_count)+'_'+str(alpha)+'_'+str(args.seed)+'_'+str(args.seed_data)
			ss_s_desc = 'mnist_SS_Sgan_'+str(args.ss_count)+'_'+str(alpha)+'_'+str(args.seed)+'_'+str(args.seed_data)

			disc_params = [sharedX(p) for p in joblib.load('models/%s/discrim_params.jl'%(ss_c_desc))]
			ss_c_dX = s_discrim(X, Y, *disc_params); ss_c_disc = theano.function([X, Y], ss_c_dX, on_unused_input='ignore')

			disc_params = [sharedX(p) for p in joblib.load('models/%s/discrim_params.jl'%(ss_ac_desc))]
			ss_ac_dX = ac_discrim(X, Y, *disc_params); ss_ac_disc = theano.function([X, Y], ss_ac_dX, on_unused_input='ignore')

			disc_params = [sharedX(p) for p in joblib.load('models/%s/discrim_params.jl'%(ss_s_desc))]
			ss_s_dX = ss_discrim(X, Y, *disc_params); ss_s_disc = theano.function([X, Y], ss_s_dX, on_unused_input='ignore')


			for idx in range(args.batch_loops):
				sample_xmb = floatX(taX[idx*nbatch:((idx+1)*nbatch)])
				sample_ymb = floatX(taY[idx*nbatch:((idx+1)*nbatch)])
				sample_ymb_C = TAY[idx*nbatch:((idx+1)*nbatch)]

				ss_s_feat = np.asarray(ss_s_disc(sample_xmb, sample_ymb))
				ss_c_feat = np.asarray(ss_c_disc(sample_xmb, sample_ymb))
				ss_ac_feat = np.asarray(ss_ac_disc(sample_xmb, sample_ymb))

				ss_s_score += metrics.accuracy_score(sample_ymb_C,cross_val_predict(logreg, ss_s_feat, sample_ymb_C, cv=3))
				ss_c_score += metrics.accuracy_score(sample_ymb_C,cross_val_predict(logreg, ss_c_feat, sample_ymb_C, cv=3))
				ss_ac_score += metrics.accuracy_score(sample_ymb_C,cross_val_predict(logreg, ss_ac_feat, sample_ymb_C, cv=3))

    args.seed_data = args.seed_data_list[0]
    args.seed = args.seed_list[0]

    c_desc = 'mnist_Cgan_'+str(args.s_count)+'_'+'alp'+'_'+str(args.seed)+'_'+str(args.seed_data)
    ac_desc = 'mnist_ACgan_'+str(args.s_count)+'_'+str(alpha)+'_'+str(args.seed)+'_'+str(args.seed_data)

    disc_params = [sharedX(p) for p in joblib.load('models/%s/discrim_params.jl'%(c_desc))]
    c_dX = s_discrim(X, Y, *disc_params); c_disc = theano.function([X, Y], c_dX, on_unused_input='ignore')

    disc_params = [sharedX(p) for p in joblib.load('models/%s/discrim_params.jl'%(ac_desc))]
    ac_dX = ac_discrim(X, Y, *disc_params); ac_disc = theano.function([X, Y], ac_dX, on_unused_input='ignore')


    for idx in range(args.batch_loops):
        sample_xmb = floatX(taX[idx*nbatch:((idx+1)*nbatch)])
        sample_ymb = floatX(taY[idx*nbatch:((idx+1)*nbatch)])
        sample_ymb_C = TAY[idx*nbatch:((idx+1)*nbatch)]

        ac_feat = np.asarray(ac_disc(sample_xmb, sample_ymb))
        c_feat = np.asarray(c_disc(sample_xmb, sample_ymb))

        c_score += metrics.accuracy_score(sample_ymb_C,cross_val_predict(logreg, c_feat, sample_ymb_C, cv=3))
        ac_score += metrics.accuracy_score(sample_ymb_C,cross_val_predict(logreg, ac_feat, sample_ymb_C, cv=3))


    ss_s_score /= len(args.seed_data_list)*len(args.seed_list)*args.batch_loops
    ss_c_score /= len(args.seed_data_list)*len(args.seed_list)*args.batch_loops
    ss_ac_score /= len(args.seed_data_list)*len(args.seed_list)*args.batch_loops
    c_score /= args.batch_loops
    ac_score /= args.batch_loops


    return ss_s_score, ss_c_score, ss_ac_score, c_score, ac_score

if __name__=='__main__':
	ss_s_score, ss_c_score, ss_ac_score, c_score, ac_score = evaluate_discrim_classifier(args)


