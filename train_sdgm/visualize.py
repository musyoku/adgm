# -*- coding: utf-8 -*-
import os, sys, time, pylab
import numpy as np
from chainer import cuda, Variable
import matplotlib.patches as mpatches
sys.path.append(os.path.split(os.getcwd())[0])
import util
import visualizer
from args import args
from model import conf, adgm

def sample_labeled_data():
	x_labeled, y_labeled, label_ids = sampler.x_and_label_data(batchsize_labeled, conf.ndim_x, conf.ndim_y, dataset_labeled, label_ids_labeled)

	# binalize
	x_labeled = util.binarize_data(x_labeled)

	x_labeled = Variable(x_labeled)
	y_labeled = Variable(y_labeled)
	label_ids = Variable(label_ids)

	return x_labeled, y_labeled, label_ids
	
def forward_one_step(num_images):
	x, y_labeled, label_ids = sample_labeled_data()
	x.to_gpu()
	a = adgm.encode_x_a(x, test=True)
	y = adgm.sample_ax_y(a, x, argmax=True, test=True)
	z = adgm.encode_axy_z(a, x, y, test=True)
	_x = adgm.decode_ayz_x(a, y, z, test=True)
	if conf.gpu_enabled:
		z.to_cpu()
		a.to_cpu()
		_x.to_cpu()
	_x = _x.data
	return a, z, _x, label_ids

try:
	os.mkdir(args.plot_dir)
except:
	pass

dataset, labels = util.load_labeled_images(args.test_image_dir)

a, z, _x, _ = forward_one_step(100)
visualizer.tile_x(_x, dir=args.plot_dir)

a, z, _x, label_ids = forward_one_step(5000)
visualizer.plot_z(z.data, dir=args.plot_dir)
visualizer.plot_z(a.data, dir=args.plot_dir, filename="a")
visualizer.plot_labeled_z(z.data, label_ids.data, dir=args.plot_dir)