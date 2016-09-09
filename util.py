# -*- coding: utf-8 -*-
import os, re, math, pylab, sys
from math import *
import numpy as np
from StringIO import StringIO
from PIL import Image
from chainer import cuda, Variable, function
from chainer.utils import type_check
from sklearn import preprocessing
import matplotlib

def load_images(image_dir, convert_to_grayscale=True):
	dataset = []
	fs = os.listdir(image_dir)
	i = 0
	for fn in fs:
		f = open("%s/%s" % (image_dir, fn), "rb")
		if convert_to_grayscale:
			img = np.asarray(Image.open(StringIO(f.read())).convert("L"), dtype=np.float32) / 255.0
		else:
			img = np.asarray(Image.open(StringIO(f.read())).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
		dataset.append(img)
		f.close()
		i += 1
		if i % 100 == 0:
			sys.stdout.write("\rloading images...({:d} / {:d})".format(i, len(fs)))
			sys.stdout.flush()
	sys.stdout.write("\n")
	return dataset

def load_labeled_images(image_dir, convert_to_grayscale=True):
	dataset = []
	labels = []
	fs = os.listdir(image_dir)
	i = 0
	for fn in fs:
		m = re.match("([0-9]+)_.+", fn)
		label = int(m.group(1))
		f = open("%s/%s" % (image_dir, fn), "rb")
		if convert_to_grayscale:
			img = np.asarray(Image.open(StringIO(f.read())).convert("L"), dtype=np.float32) / 255.0
		else:
			img = np.asarray(Image.open(StringIO(f.read())).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
		dataset.append(img)
		labels.append(label)
		f.close()
		i += 1
		if i % 100 == 0:
			sys.stdout.write("\rloading images...({:d} / {:d})".format(i, len(fs)))
			sys.stdout.flush()
	sys.stdout.write("\n")
	return dataset, labels

def create_semisupervised(dataset, labels, num_validation_data=10000, num_labeled_data=100, num_types_of_label=10):
	if len(dataset) < num_validation_data + num_labeled_data:
		raise Exception("len(dataset) < num_validation_data + num_labeled_data")
	training_labeled_x = []
	training_unlabeled_x = []
	validation_x = []
	validation_labels = []
	training_labels = []
	indices_for_label = {}
	num_data_per_label = int(num_labeled_data / num_types_of_label)
	num_unlabeled_data = len(dataset) - num_validation_data - num_labeled_data

	indices = np.arange(len(dataset))
	np.random.shuffle(indices)

	def check(index):
		label = labels[index]
		if label not in indices_for_label:
			indices_for_label[label] = []
			return True
		if len(indices_for_label[label]) < num_data_per_label:
			for i in indices_for_label[label]:
				if i == index:
					return False
			return True
		return False

	for n in xrange(len(dataset)):
		index = indices[n]
		if check(index):
			indices_for_label[labels[index]].append(index)
			training_labeled_x.append(dataset[index])
			training_labels.append(labels[index])
		else:
			if len(training_unlabeled_x) < num_unlabeled_data:
				training_unlabeled_x.append(dataset[index])
			else:
				validation_x.append(dataset[index])
				validation_labels.append(labels[index])

	return training_labeled_x, training_labels, training_unlabeled_x, validation_x, validation_labels

def binarize_data(x):
	threshold = np.random.uniform(size=x.shape)
	return np.where(threshold < x, 1.0, 0.0).astype(np.float32)