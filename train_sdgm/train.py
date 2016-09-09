# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
import pandas as pd
sys.path.append(os.path.split(os.getcwd())[0])
import util
from args import args
from model import conf, adgm

dataset, labels = util.load_labeled_images(args.train_image_dir, dist=conf.distribution_x)

max_epoch = 1000
n_trains_per_epoch = 300
batchsize_l = 100
batchsize_u = 200

# Create labeled/unlabeled split in training set
n_types_of_label = 10
n_labeled_data = args.n_labeled_data
n_validation_data = args.n_validation_data
labeled_dataset, labels, unlabeled_dataset, validation_dataset, validation_labels = util.create_semisupervised(dataset, labels, n_validation_data, n_labeled_data, n_types_of_label)
print "labels:", labels
alpha = 0.1 * len(dataset) / len(labeled_dataset)
alpha = 1.0
print "alpha:", alpha
print "dataset:: labeled: {} unlabeled: {} validation: {}".format(len(labeled_dataset), len(unlabeled_dataset), len(validation_dataset))

if n_labeled_data < batchsize_l:
	batchsize_l = n_labeled_data
	
if len(unlabeled_dataset) < batchsize_u:
	batchsize_u = len(unlabeled_dataset)

# Export result to csv
csv_epoch = []

total_time = 0
for epoch in xrange(max_epoch):
	sum_loss_labeled = 0
	sum_loss_unlabeled = 0
	sum_loss_classifier = 0
	epoch_time = time.time()
	for t in xrange(n_trains_per_epoch):
		x_labeled, y_labeled, label_ids = util.sample_x_and_label_variables(batchsize_l, conf.ndim_x, conf.ndim_y, labeled_dataset, labels, gpu_enabled=conf.gpu_enabled)
		x_unlabeled = util.sample_x_variable(batchsize_u, conf.ndim_x, unlabeled_dataset, gpu_enabled=conf.gpu_enabled)

		# train
		loss_labeled, loss_unlabeled = adgm.train(x_labeled, y_labeled, label_ids, x_unlabeled)
		loss_classifier = adgm.train_classification(x_labeled, label_ids, alpha=alpha)

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
	adgm.save(args.model_dir)

	# validation
	x_labeled, _, label_ids = util.sample_x_and_label_variables(n_validation_data, conf.ndim_x, conf.ndim_y, validation_dataset, validation_labels, gpu_enabled=False)
	if conf.gpu_enabled:
		x_labeled.to_gpu()
	prediction = adgm.sample_x_label(x_labeled, test=True, argmax=True)
	correct = 0
	for i in xrange(n_validation_data):
		if prediction[i] == label_ids.data[i]:
			correct += 1
	print "classification accuracy (validation): {}".format(correct / float(n_validation_data))

	# Export to csv
	csv_epoch.append([epoch, int(total_time / 60), correct / float(n_validation_data)])
	data = pd.DataFrame(csv_epoch)
	data.columns = ["epoch", "min", "accuracy"]
	data.to_csv("{}/epoch.csv".format(args.model_dir))

