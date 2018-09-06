#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:27:51 2017

@author: skumar
"""

import sys
sys.path.append('..')

import argparse

from train_SS_C_GAN import SS_C_GAN
from train_SS_AC_GAN import SS_AC_GAN
from train_SS_S_GAN import SS_S_GAN
from train_C_GAN import C_GAN
from train_AC_GAN import AC_GAN

from train_attribute_predictor import attribute_predictor
from train_sample_predictor import sample_predictor

from evaluate_attribute_prediction import  evaluate_attribute_prediction
from evaluate_sample_error import evaluate_sample_prediction
from evaluate_reconstruction_error import evaluate_reconstruction_error
from evaluate_discriminator_classif import evaluate_discrim_classifier

parser = argparse.ArgumentParser()
parser.add_argument('--seed_list', default=[1])
parser.add_argument('--seed_data_list', default=[1,2,3,4,5,6])
parser.add_argument('--ss_count', default=50)
parser.add_argument('--s_count', default=50000)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--discrim_batch_size', default=5000)
parser.add_argument('--labeled_weight', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--niter', default=100)
parser.add_argument('--batch_loops', default=10)
parser.add_argument('--adam_recon_loops', default=500)
args = parser.parse_args()
    
def train_models(args):

	#train all fully supervised models once
	args.seed_data = args.seed_data_list[0]
	args.seed = args.seed_list[0]
	args.count = args.s_count
	C_GAN(args)
	AC_GAN(args)			
	attribute_predictor(args)

	#train semi-supervised models multiple times
	
	for sd in args.seed_data_list:
		for s in args.seed_list:
			args.seed_data = sd
			args.seed = s

			args.count = args.ss_count
			SS_C_GAN(args)
			SS_S_GAN(args)
			SS_AC_GAN(args)
			sample_predictor(args)


if __name__=='__main__':
	print "Start of evaluation"
	train_models(args)
	print "Model training completed"
	ap_true_score, ap_ss_s_score, ap_ss_c_score, ap_ss_ac_score, ap_c_score, ap_ac_score = evaluate_attribute_prediction(args)
	print "Attribute prediction evaluation completed"
	re_ss_s_score, re_ss_c_score, re_ss_ac_score, re_c_score, re_ac_score = evaluate_reconstruction_error(args)
	print "Reconstruction error evaluation completed"
	sp_true_score, sp_fake_score, sp_ss_s_score, sp_ss_c_score, sp_ss_ac_score, sp_c_score, sp_ac_score = evaluate_sample_prediction(args)
	print "Sample prediction evaluation completed"
	dc_ss_s_score, dc_ss_c_score, dc_ss_ac_score, dc_c_score, dc_ac_score = evaluate_discrim_classifier(args)
	print "Discriminator classification evaluation completed"
	

