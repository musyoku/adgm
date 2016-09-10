# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
import pandas as pd
sys.path.append(os.path.split(os.getcwd())[0])
import util, sampler
from args import args
from model import conf, adgm

def sample_labeled_data():
	x_labeled = Variable(np.asarray([[-1, 0, 1], [1, 0, -1]], dtype=np.float32))
	y_labeled = Variable(np.asarray([[1, 0], [0, 1]], dtype=np.float32))
	label_ids = Variable(np.asarray([0, 1], dtype=np.int32))

	if conf.gpu_enabled:
		x_labeled.to_gpu()
		y_labeled.to_gpu()
		label_ids.to_gpu()

	return x_labeled, y_labeled, label_ids

def sample_unlabeled_data():
	x_unlabeled = Variable(np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32))

	if conf.gpu_enabled:
		x_unlabeled.to_gpu()

	return x_unlabeled

max_epoch = 1000
n_trains_per_epoch = 500
n_types_of_label = conf.ndim_y
n_labeled_data = args.n_labeled_data
n_validation_data = 0
batchsize_labeled = 2
batchsize_unlabeled = 3

# seed
np.random.seed(args.seed)
if conf.gpu_enabled:
    cuda.cupy.random.seed(args.seed)

total_time = 0
for epoch in xrange(max_epoch):
	sum_loss_labeled = 0
	sum_loss_unlabeled = 0
	sum_loss_classifier = 0
	epoch_time = time.time()

	for t in xrange(n_trains_per_epoch):
		# sample labeled data
		x_labeled, y_labeled, label_ids = sample_labeled_data()

		# sample unlabeled data
		x_unlabeled = sample_unlabeled_data()

		# train
		loss_labeled, loss_unlabeled = adgm.train(x_labeled, y_labeled, label_ids, x_unlabeled)
		loss_classifier = adgm.train_classification(x_labeled, label_ids)

		sum_loss_labeled += loss_labeled
		sum_loss_unlabeled += loss_unlabeled
		sum_loss_classifier += loss_classifier
		if t % 10 == 0:
			sys.stdout.write("\rTraining in progress...({} / {})".format(t, n_trains_per_epoch))
			sys.stdout.flush()

	epoch_time = time.time() - epoch_time
	total_time += epoch_time
	sys.stdout.write("\r")
	print "epoch: {} loss:: labeled: {:.3f} unlabeled: {:.3f} classifier: {:.3f} time: {} min total: {} min".format(epoch + 1, sum_loss_labeled / n_trains_per_epoch, sum_loss_unlabeled / n_trains_per_epoch, sum_loss_classifier / n_trains_per_epoch, int(epoch_time / 60), int(total_time / 60))
	sys.stdout.flush()
