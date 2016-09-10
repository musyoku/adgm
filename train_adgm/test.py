# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
import pandas as pd
sys.path.append(os.path.split(os.getcwd())[0])
import util, sampler
from args import args
from model import conf, adgm

def sample_test_data():
	x_labeled, _, label_ids = sampler.x_and_label_data(n_test_data, conf.ndim_x, conf.ndim_y, dataset_all, label_ids_all)

	# binalize
	x_labeled = util.binarize_data(x_labeled)

	x_labeled = Variable(x_labeled)
	label_ids = Variable(label_ids)

	if conf.gpu_enabled:
		x_labeled.to_gpu()

	return x_labeled, label_ids

# load all images
dataset_all, label_ids_all = util.load_labeled_images(args.test_image_dir)
n_test_data = len(dataset_all)

# test
x_labeled, label_ids = sample_test_data()
predicted_ids = adgm.sample_x_label(x_labeled, test=True, argmax=True)
correct = 0
for i in xrange(n_test_data):
	if predicted_ids[i] == label_ids.data[i]:
		correct += 1
print "classification accuracy (test): {}".format(correct / float(n_test_data))

