#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 16:17:33 2016

@author: zhaowenzhi
"""
import sample_extraction as samples # sample_extraction.py
import pickle
import os
from scipy import misc

############################
#Parameter setting
tr_number = 0.1 # 
te_number = 0.1 #
patchshape = [28,28]#

#Set the original image and reference map
img_path = os.path.join(os.getcwd(), 'ori_3_shrink.tif')
gt_path = os.path.join(os.getcwd(), 'gt_3_shrink.tif')

############################
img = misc.imread(img_path)
gt = misc.imread(gt_path,0)

#Assign training variables 28x28x3 input img
[trX,trY_ori,tr_corrs] = samples.extract_samples(img,gt,patchshape,tr_number)
[teX,teY_ori,te_corrs] = samples.extract_samples(img,gt,patchshape,te_number)

training_samples_path = os.path.join(os.getcwd(), 'samples_data/training_patches_ori3.ckpt')
training_labels_path = os.path.join(os.getcwd(), 'samples_data/training_labels_ori3.ckpt')
test_samples_path = os.path.join(os.getcwd(), 'samples_data/test_patches_ori3.ckpt')
test_labels_path = os.path.join(os.getcwd(), 'samples_data/test_labels_ori3.ckpt')

output_1 = open(training_samples_path, 'wb')
pickle.dump(trX, output_1)
output_1.close()
output_2 = open(training_labels_path, 'wb')
pickle.dump(trY_ori, output_2)
output_2.close()
output_3 = open(test_samples_path, 'wb')
pickle.dump(teX, output_3)
output_3.close()
output_4 = open(test_labels_path, 'wb')
pickle.dump(teY_ori, output_4)
output_4.close()



