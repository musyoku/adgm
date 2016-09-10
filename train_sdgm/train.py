# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
import pandas as pd
sys.path.append(os.path.split(os.getcwd())[0])
import util, sampler
from args import args
from model import conf, sdgm

def sample_labeled_data():
	x_labeled, y_labeled, label_ids = sampler.x_and_label_data(batchsize_labeled, conf.ndim_x, conf.ndim_y, dataset_labeled, label_ids_labeled)

	# binalize
	x_labeled = util.binarize_data(x_labeled)

	x_labeled = Variable(x_labeled)
	y_labeled = Variable(y_labeled)
	label_ids = Variable(label_ids)

	if conf.gpu_enabled:
		x_labeled.to_gpu()
		y_labeled.to_gpu()
		label_ids.to_gpu()

	return x_labeled, y_labeled, label_ids

def sample_validation_data():
	x_labeled, _, label_ids = sampler.x_and_label_data(n_validation_data, conf.ndim_x, conf.ndim_y, dataset_validation, label_ids_validation)

	# binalize
	x_labeled = util.binarize_data(x_labeled)

	x_labeled = Variable(x_labeled)
	label_ids = Variable(label_ids)

	if conf.gpu_enabled:
		x_labeled.to_gpu()

	return x_labeled, label_ids

def sample_unlabeled_data():
	x_unlabeled = sampler.x_data(batchsize_unlabeled, conf.ndim_x, dataset_unlabeld)

	# binalize
	x_unlabeled = util.binarize_data(x_unlabeled)

	x_unlabeled = Variable(x_unlabeled)
	if conf.gpu_enabled:
		x_unlabeled.to_gpu()

	return x_unlabeled

max_epoch = 1000
n_trains_per_epoch = 500
n_types_of_label = 10
n_labeled_data = args.n_labeled_data
n_validation_data = args.n_validation_data
batchsize_labeled = 100
batchsize_unlabeled = 200

# export result to csv
csv_epochs = []

# load all images
dataset_all, label_ids_all = util.load_labeled_images(args.train_image_dir)

# Create labeled/unlabeled split in training set
dataset_labeled, label_ids_labeled, dataset_unlabeld, dataset_validation, label_ids_validation = util.create_semisupervised(dataset_all, label_ids_all, n_validation_data, n_labeled_data, n_types_of_label, seed=args.seed_ssl_split)
print "labels for supervised training:", label_ids_labeled
# alpha = 0.1 * len(dataset) / len(dataset_labeled)
alpha = 1.0
print "alpha:", alpha
print "dataset:: labeled: {} unlabeled: {} validation: {}".format(len(dataset_labeled), len(dataset_unlabeld), len(dataset_validation))

# fix
if n_labeled_data < batchsize_labeled:
	batchsize_labeled = n_labeled_data
	
if len(dataset_unlabeld) < batchsize_unlabeled:
	batchsize_unlabeled = len(dataset_unlabeld)

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
		loss_labeled, loss_unlabeled = sdgm.train(x_labeled, y_labeled, label_ids, x_unlabeled)
		loss_classifier = sdgm.train_classification(x_labeled, label_ids, alpha=alpha)

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
	sdgm.save(args.model_dir)

	# validation
	x_labeled, label_ids = sample_validation_data()
	predicted_ids = sdgm.sample_x_label(x_labeled, test=True, argmax=True)
	correct = 0
	for i in xrange(n_validation_data):
		if predicted_ids[i] == label_ids.data[i]:
			correct += 1
	print "classification accuracy (validation): {}".format(correct / float(n_validation_data))

	# export to csv
	csv_epochs.append([epoch, int(total_time / 60), correct / float(n_validation_data)])
	data = pd.DataFrame(csv_epochs)
	data.columns = ["epoch", "min", "accuracy"]
	data.to_csv("{}/epoch.csv".format(args.model_dir))

