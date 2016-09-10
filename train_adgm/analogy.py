# -*- coding: utf-8 -*-
import os, sys, time, pylab
import numpy as np
from chainer import cuda, Variable
import matplotlib.patches as mpatches
sys.path.append(os.path.split(os.getcwd())[0])
import util, sampler
from args import args
from model import conf, adgm

try:
	os.mkdir(args.plot_dir)
except:
	pass

# load all images
dataset_all = util.load_images(args.test_image_dir)

n_analogies = 10
n_image_channels = 1
image_width = 28
image_height = 28

# sample data
x = sampler.x_data(10, conf.ndim_x, dataset_all)
x = Variable(x)
if conf.gpu_enabled:
	x.to_gpu()
z = adgm.encode_x_z(x, argmax=True, test=True)

fig = pylab.gcf()
fig.set_size_inches(16.0, 16.0)
pylab.clf()
if n_image_channels == 1:
	pylab.gray()
xp = np
if conf.gpu_enabled:
	x.to_cpu()
	xp = cuda.cupy
for m in xrange(n_analogies):
	pylab.subplot(n_analogies, conf.ndim_y + 2, m * 12 + 1)
	if n_image_channels == 1:
		pylab.imshow(x.data[m].reshape((image_width, image_height)), interpolation="none")
	elif n_image_channels == 3:
		pylab.imshow(x.data[m].reshape((n_image_channels, image_width, image_height)), interpolation="none")
	pylab.axis("off")
all_y = xp.identity(conf.ndim_y, dtype=xp.float32)
all_y = Variable(all_y)
for m in xrange(n_analogies):
	base_z = xp.empty((conf.ndim_y, z.data.shape[1]), dtype=xp.float32)
	for n in xrange(conf.ndim_y):
		base_z[n] = z.data[m]
	base_z = Variable(base_z)
	_x = adgm.decode_yz_x(all_y, base_z, test=True, apply_f=True)
	if conf.gpu_enabled:
		_x.to_cpu()
	for n in xrange(conf.ndim_y):
		pylab.subplot(n_analogies, conf.ndim_y + 2, m * 12 + 3 + n)
		if n_image_channels == 1:
			pylab.imshow(_x.data[n].reshape((image_width, image_height)), interpolation="none")
		elif n_image_channels == 3:
			pylab.imshow(_x.data[n].reshape((n_image_channels, image_width, image_height)), interpolation="none")
		pylab.axis("off")

pylab.savefig("{:s}/analogy.png".format(args.plot_dir))

