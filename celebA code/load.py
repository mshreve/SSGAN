# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:57:39 2016

@author: skumar
"""

import sys
sys.path.append('..')

from PIL import Image
import glob
import numpy as np

def faces(batch_size=128):
    
    f = open('datalabels/list_attr_celeba.txt','r')
    attr_list_global = f.readlines()
    f.close()
    
    attr_list_global_name = [x.split()[0] for x in attr_list_global]
    attr_dict = dict([(x,i) for (i,x) in enumerate(attr_list_global_name)])
    
    attr_list_global = [x.split()[1:] for x in attr_list_global]
    attr_list_global = np.array(attr_list_global).astype('int8')
    attr_subset = np.array([4,5,8,9,11,12,15,17,18,20,21,22,26,28,31,32,33,35])
    attr_list_global = (attr_list_global[:,attr_subset]>0.).astype('int8')
    
        
    image_list = []
    attr_list = []
    i = 0
    for filename in glob.glob('celebAsmall/*.jpg'): #assuming gif
        im=Image.open(filename)
        im = im.resize((im.size[0]/2,im.size[1]/2))
        image_list.append(np.array(im).reshape(im.size[1], im.size[0], 3))        
        im.close()
        attr_list.append(attr_list_global[attr_dict[filename.split('/')[1]]])
        
        
    return image_list, attr_list