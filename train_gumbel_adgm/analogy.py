# -*- coding: utf-8 -*-
import os, sys, time, pylab
import numpy as np
from chainer import cuda, Variable
import matplotlib.patches as mpatches
sys.path.append(os.path.split(os.getcwd())[0])
import dataset
from model import adgm
from args import args

try:
	os.mkdir(args.plot_dir)
except:
	pass

# load test images
images, labels = dataset.load_test_images()

# config
config = adgm.config
num_analogies = 10
xp = np
if args.gpu_device != -1:
	xp = cuda.cupy

# sample data
x = dataset.sample_unlabeled_data(images, num_analogies, config.ndim_x, binarize=False)
z = adgm.encode_x_z(x, argmax_y=True, test=True)

# plot
fig = pylab.gcf()
fig.set_size_inches(16.0, 16.0)
pylab.clf()
pylab.gray()
for m in xrange(num_analogies):
	pylab.subplot(num_analogies, config.ndim_y + 2, m * 12 + 1)
	pylab.imshow(x[m].reshape((28, 28)), interpolation="none")
	pylab.axis("off")
all_y = xp.identity(config.ndim_y, dtype=xp.float32)
for m in xrange(num_analogies):
	fixed_z_repeat = xp.repeat(z.data[m].reshape((1, -1)), config.ndim_y, axis=0)
	_x = adgm.decode_yz_x(all_y, fixed_z_repeat, test=True)
	if args.gpu_device != -1:
		_x.to_cpu()
	for n in xrange(config.ndim_y):
		pylab.subplot(num_analogies, config.ndim_y + 2, m * 12 + 3 + n)
		pylab.imshow(_x.data[n].reshape((28, 28)), interpolation="none")
		pylab.axis("off")

pylab.savefig("{}/analogy.png".format(args.plot_dir))

