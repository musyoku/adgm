# -*- coding: utf-8 -*-
import math
import json, os, sys
from chainer import cuda
from args import args
sys.path.append(os.path.split(os.getcwd())[0])
from adgm import SDGM, Config
from sequential import Sequential
from sequential.link import Linear, Merge, BatchNormalization, Gaussian
from sequential.function import Activation, dropout, gaussian_noise, tanh, sigmoid

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass

# specify energy model
model_filename = args.model_dir + "/model.json"

if os.path.isfile(model_filename):
	print "loading", model_filename
	with open(model_filename, "r") as f:
		try:
			params = json.load(f)
		except Exception as e:
			raise Exception("could not load {}".format(model_filename))
else:
	config = Config()
	config.ndim_a = 100
	config.ndim_x = 28 * 28
	config.ndim_y = 10
	config.ndim_z = 100
	config.weight_init_std = 0.01
	config.weight_initializer = "Normal"
	config.nonlinearity = "elu"
	config.optimizer = "Eve"
	config.learning_rate = 0.0003
	config.momentum = 0.9
	config.gradient_clipping = 10
	config.weight_decay = 0
	config.use_weightnorm = False
	config.num_mc_samples = 1

	# p(x|y,z) - x ~ Bernoulli
	p_x_ayz = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	p_x_ayz.add(Merge(num_inputs=3, out_size=500, use_weightnorm=config.use_weightnorm))
	p_x_ayz.add(BatchNormalization(500))
	p_x_ayz.add(Activation(config.nonlinearity))
	p_x_ayz.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	p_x_ayz.add(BatchNormalization(500))
	p_x_ayz.add(Activation(config.nonlinearity))
	p_x_ayz.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	p_x_ayz.add(BatchNormalization(500))
	p_x_ayz.add(Activation(config.nonlinearity))
	p_x_ayz.add(Linear(None, config.ndim_x, use_weightnorm=config.use_weightnorm))
	p_x_ayz.build()

	# p(a|x,y,z) - a ~ Gaussian
	p_a_yz = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	p_a_yz.add(Merge(num_inputs=2, out_size=500, use_weightnorm=config.use_weightnorm))
	p_a_yz.add(BatchNormalization(500))
	p_a_yz.add(Activation(config.nonlinearity))
	p_a_yz.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	p_a_yz.add(BatchNormalization(500))
	p_a_yz.add(Activation(config.nonlinearity))
	p_a_yz.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	p_a_yz.add(BatchNormalization(500))
	p_a_yz.add(Activation(config.nonlinearity))
	p_a_yz.add(Gaussian(None, config.ndim_a))	# outputs mean and ln(var)
	p_a_yz.build()

	# q(z|a,x,y) - z ~ Gaussian
	q_z_axy = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	q_z_axy.add(Merge(num_inputs=3, out_size=500, use_weightnorm=config.use_weightnorm))
	q_z_axy.add(BatchNormalization(500))
	q_z_axy.add(Activation(config.nonlinearity))
	q_z_axy.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	q_z_axy.add(BatchNormalization(500))
	q_z_axy.add(Activation(config.nonlinearity))
	q_z_axy.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	q_z_axy.add(BatchNormalization(500))
	q_z_axy.add(Activation(config.nonlinearity))
	q_z_axy.add(Gaussian(None, config.ndim_z))	# outputs mean and ln(var)
	q_z_axy.build()

	# q(a|x) - a ~ Gaussian
	q_a_x = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	q_a_x.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	q_a_x.add(BatchNormalization(500))
	q_a_x.add(Activation(config.nonlinearity))
	q_a_x.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	q_a_x.add(BatchNormalization(500))
	q_a_x.add(Activation(config.nonlinearity))
	q_a_x.add(Gaussian(None, config.ndim_a))	# outputs mean and ln(var)
	q_a_x.build()

	# q(y|a,x) - y ~ Categorical
	q_y_ax = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	q_y_ax.add(Merge(num_inputs=2, out_size=500, use_weightnorm=config.use_weightnorm))
	q_y_ax.add(BatchNormalization(500))
	q_y_ax.add(Activation(config.nonlinearity))
	q_y_ax.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	q_y_ax.add(BatchNormalization(500))
	q_y_ax.add(Activation(config.nonlinearity))
	q_y_ax.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	q_y_ax.add(BatchNormalization(500))
	q_y_ax.add(Activation(config.nonlinearity))
	q_y_ax.add(Linear(None, config.ndim_y, use_weightnorm=config.use_weightnorm))
	q_y_ax.build()

	params = {
		"config": config.to_dict(),
		"p_a_yz": p_a_yz.to_dict(),
		"p_x_ayz": p_x_ayz.to_dict(),
		"q_a_x": q_a_x.to_dict(),
		"q_y_ax": q_y_ax.to_dict(),
		"q_z_axy": q_z_axy.to_dict(),
	}

	with open(model_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

sdgm = SDGM(params)
sdgm.load(args.model_dir)

if args.gpu_device != -1:
	cuda.get_device(args.gpu_device).use()
	sdgm.to_gpu()
