# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six
from chainer import cuda, Variable, optimizers, serializers, function, optimizer
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
from adgm import activations, DGM, MultiLayerPerceptron, GaussianEncoder, GradientClipping

class Conf():
	def __init__(self):
		self.image_width = 28
		self.image_height = 28
		self.ndim_x = 28 * 28
		self.distribution_x = "bernoulli"
		self.ndim_y = 10
		self.ndim_z = 100
		self.ndim_a = 100
		self.n_mc_samples = 10
		self.wscale = 0.001

		# True : y = f(BN(Wx + b))
		# False: y = f(W*BN(x) + b)
		self.batchnorm_before_activation = True

		self.encoder_x_a_hidden_units = [500, 500]
		self.encoder_x_a_activation_function = "elu"
		self.encoder_x_a_apply_dropout = False
		self.encoder_x_a_apply_batchnorm = True
		self.encoder_x_a_apply_batchnorm_to_input = True

		self.encoder_xy_z_hidden_units = [500, 500]
		self.encoder_xy_z_activation_function = "elu"
		self.encoder_xy_z_apply_dropout = False
		self.encoder_xy_z_apply_batchnorm = True
		self.encoder_xy_z_apply_batchnorm_to_input = True

		self.encoder_ax_y_hidden_units = [500, 500]
		self.encoder_ax_y_activation_function = "elu"
		self.encoder_ax_y_apply_dropout = False
		self.encoder_ax_y_apply_batchnorm = True
		self.encoder_ax_y_apply_batchnorm_to_input = True

		self.decoder_ayz_x_hidden_units = [500, 500]
		self.decoder_ayz_x_activation_function = "elu"
		self.decoder_ayz_x_apply_dropout = False
		self.decoder_ayz_x_apply_batchnorm = True
		self.decoder_ayz_x_apply_batchnorm_to_input = True

		self.decoder_xyz_a_hidden_units = [500, 500]
		self.decoder_xyz_a_activation_function = "elu"
		self.decoder_xyz_a_apply_dropout = False
		self.decoder_xyz_aapply_batchnorm = True
		self.decoder_xyz_a_apply_batchnorm_to_input = True

		self.gpu_enabled = True
		self.learning_rate = 0.0003
		self.gradient_momentum = 0.9
		self.gradient_clipping = 10.0

	def check(self):
		base = Conf()
		for attr, value in self.__dict__.iteritems():
			if not hasattr(base, attr):
				raise Exception("invalid parameter '{}'".format(attr))

class SDGM(DGM):
	# name is used for the filename when you save the model
	def __init__(self, conf, name="adgm"):
		conf.check()
		self.conf = conf
		self.name = name

		# q(z|a, x, y)
		self.encoder_axy_z = self.build_encoder_axy_z()
		self.optimizer_encoder_axy_z = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_encoder_axy_z.setup(self.encoder_axy_z)
		# self.optimizer_encoder_axy_z.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_encoder_axy_z.add_hook(GradientClipping(conf.gradient_clipping))

		# q(y|a, x)
		self.encoder_ax_y = self.build_encoder_ax_y()
		self.optimizer_encoder_ax_y = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_encoder_ax_y.setup(self.encoder_ax_y)
		# self.optimizer_encoder_ax_y.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_encoder_ax_y.add_hook(GradientClipping(conf.gradient_clipping))

		# q(a|x)
		self.encoder_x_a = self.build_encoder_x_a()
		self.optimizer_encoder_x_a = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_encoder_x_a.setup(self.encoder_x_a)
		# self.optimizer_encoder_x_a.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_encoder_x_a.add_hook(GradientClipping(conf.gradient_clipping))

		# p(x|y, z)
		self.decoder_ayz_x = self.build_decoder_ayz_x()
		self.optimizer_decoder_ayz_x = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_decoder_ayz_x.setup(self.decoder_ayz_x)
		# self.optimizer_decoder_ayz_x.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_decoder_ayz_x.add_hook(GradientClipping(conf.gradient_clipping))

		# p(a|x, y, z)
		self.decoder_xyz_a = self.build_decoder_xyz_a()
		self.optimizer_decoder_xyz_a = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_decoder_xyz_a.setup(self.decoder_xyz_a)
		# self.optimizer_decoder_xyz_a.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_decoder_xyz_a.add_hook(GradientClipping(conf.gradient_clipping))

	def build_decoder_ayz_x(self):
		conf = self.conf
		attributes = {}
		units = zip(conf.decoder_ayz_x_hidden_units[:-1], conf.decoder_ayz_x_hidden_units[1:])
		units += [(conf.decoder_ayz_x_hidden_units[-1], conf.ndim_x)]
		for i, (n_in, n_out) in enumerate(units):
			attributes["layer_%i" % i] = L.Linear(n_in, n_out, initialW=np.random.normal(scale=conf.wscale, size=(n_out, n_in)))
			if conf.batchnorm_before_activation:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		attributes["layer_merge_a"] = L.Linear(conf.ndim_a, conf.decoder_ayz_x_hidden_units[0], initialW=np.random.normal(scale=conf.wscale, size=(conf.decoder_ayz_x_hidden_units[0], conf.ndim_a)))
		attributes["layer_merge_z"] = L.Linear(conf.ndim_z, conf.decoder_ayz_x_hidden_units[0], initialW=np.random.normal(scale=conf.wscale, size=(conf.decoder_ayz_x_hidden_units[0], conf.ndim_z)))
		attributes["layer_merge_y"] = L.Linear(conf.ndim_y, conf.decoder_ayz_x_hidden_units[0], initialW=np.random.normal(scale=conf.wscale, size=(conf.decoder_ayz_x_hidden_units[0], conf.ndim_y)))

		if conf.batchnorm_before_activation:
			attributes["batchnorm_merge"] = L.BatchNormalization(conf.decoder_ayz_x_hidden_units[0])
		else:
			attributes["batchnorm_merge_a"] = L.BatchNormalization(conf.ndim_x)
			attributes["batchnorm_merge_z"] = L.BatchNormalization(conf.ndim_z)

		if conf.distribution_x == "bernoulli":
			decoder_ayz_x = BernoulliDecoder_AYZ_X(**attributes)
		else:
			decoder_ayz_x = GaussianDecoder_AYZ_X(**attributes)
		decoder_ayz_x.n_layers = len(units)
		decoder_ayz_x.activation_function = conf.decoder_ayz_x_activation_function
		decoder_ayz_x.apply_dropout = conf.decoder_ayz_x_apply_dropout
		decoder_ayz_x.apply_batchnorm = conf.decoder_ayz_x_apply_batchnorm
		decoder_ayz_x.apply_batchnorm_to_input = conf.decoder_ayz_x_apply_batchnorm_to_input
		decoder_ayz_x.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			decoder_ayz_x.to_gpu()

		return decoder_ayz_x

	def zero_grads(self):
		self.optimizer_encoder_axy_z.zero_grads()
		self.optimizer_encoder_ax_y.zero_grads()
		self.optimizer_encoder_x_a.zero_grads()
		self.optimizer_decoder_ayz_x.zero_grads()
		self.optimizer_decoder_xyz_a.zero_grads()

	def update(self):
		self.optimizer_encoder_axy_z.update()
		self.optimizer_encoder_ax_y.update()
		self.optimizer_encoder_x_a.update()
		self.optimizer_decoder_ayz_x.update()
		self.optimizer_decoder_xyz_a.update()

	def decode_ayz_x(self, a, y, z, test=False, apply_f=True):
		return self.decoder_ayz_x(a, y, z, test=test, apply_f=apply_f)

	def log_px(self, a, x, y, z, test=False):
		if isinstance(self.decoder_ayz_x, BernoulliDecoder_AYZ_X):
			# do not apply F.sigmoid to the output of the decoder
			raw_output = self.decoder_ayz_x(a, y, z, test=test, apply_f=False)
			negative_log_likelihood = self.bernoulli_nll_keepbatch(x, raw_output)
			log_px_yz = -negative_log_likelihood
		else:
			x_mean, x_ln_var = self.decoder_ayz_x(a, y, z, test=test, apply_f=False)
			negative_log_likelihood = self.gaussian_nll_keepbatch(x, x_mean, x_ln_var)
			log_px_yz = -negative_log_likelihood
		return log_px_yz

class BernoulliDecoder_AYZ_X(MultiLayerPerceptron):

	def forward_one_step(self, a, y, z):
		f = activations[self.activation_function]
		if self.apply_batchnorm_to_input:
			if self.batchnorm_before_activation:
				merged_input = f(self.batchnorm_merge(self.layer_merge_z(z) + self.layer_merge_y(y) + self.layer_merge_a(a), test=self.test))
			else:
				merged_input = f(self.layer_merge_z(self.batchnorm_merge(z, test=self.test)) + self.layer_merge_y(y) + self.layer_merge_a(self.batchnorm_merge(a, test=self.test)))
		else:
			merged_input = f(self.layer_merge_z(z) + self.layer_merge_y(y) + self.layer_merge_a(a))
		return self.compute_output(merged_input)

	def __call__(self, a, y, z, test=False, apply_f=False):
		self.test = test
		output = self.forward_one_step(a, y, z)
		if apply_f:
			return F.sigmoid(output)
		return output

class GaussianDecoder_AYZ_X(GaussianEncoder):

	def merge_input(self, a, y, z):
		f = activations[self.activation_function]
		if self.apply_batchnorm_to_input:
			if self.batchnorm_before_activation:
				merged_input = f(self.batchnorm_merge(self.layer_merge_z(z) + self.layer_merge_y(y) + self.layer_merge_a(a), test=self.test))
			else:
				merged_input = f(self.layer_merge_z(self.batchnorm_merge_z(z, test=self.test)) + self.layer_merge_y(y) + self.layer_merge_a(self.batchnorm_merge_a(a, test=self.test)))
		else:
			merged_input = f(self.layer_merge_z(z) + self.layer_merge_y(y) + self.layer_merge_a(a))

		return merged_input

	def forward_one_step(self, a, y, z):
		merged_input = self.merge_input(a, y, z)
		return self.compute_output(merged_input)

	def __call__(self, a, y, z, test=False, apply_f=False):
		self.test = test
		mean, ln_var = self.forward_one_step(a, y, z)
		if apply_f:
			return F.gaussian(mean, ln_var)
		return mean, ln_var