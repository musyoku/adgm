# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six, math, random, copy
from chainer import cuda, Variable, serializers
from chainer import functions as F
import params
import sequential

class Object(object):
	pass

def to_object(dict):
	obj = Object()
	for key, value in dict.iteritems():
		setattr(obj, key, value)
	return obj

class Config(params.Params):
	def __init__(self):
		self.ndim_a = 100
		self.ndim_x = 28 * 28
		self.ndim_z = 100
		self.ndim_y = 10
		self.weight_init_std = 0.01
		self.weight_initializer = "Normal"		# Normal or GlorotNormal or HeNormal
		self.nonlinearity = "elu"
		self.optimizer = "Adam"
		self.learning_rate = 0.0001
		self.momentum = 0.9
		self.gradient_clipping = 10
		self.weight_decay = 0
		self.use_weightnorm = False
		self.num_mc_samples = 1

class DGM(object):
	def __init__(self, params):
		self.params = copy.deepcopy(params)
		self.config = to_object(params["config"])
		self._gpu = False

	@property
	def gpu_enabled(self):
		if cuda.available is False:
			return False
		return self._gpu

	@property
	def xp(self):
		if self.gpu_enabled:
			return cuda.cupy
		return np

	def to_variable(self, x):
		if isinstance(x, Variable) == False:
			x = Variable(x)
			if self.gpu_enabled:
				x.to_gpu()
		return x

	def to_numpy(self, x):
		if isinstance(x, Variable) == True:
			x.to_cpu()
			x = x.data
		if isinstance(x, cuda.ndarray) == True:
			x = cuda.to_cpu(x)
		return x

	def get_batchsize(self, x):
		return x.shape[0]

	def encode_x_a(self, x, test=False):
		x = self.to_variable(x)
		mean, ln_var = self.q_a_x(x, test=test)
		return F.gaussian(mean, ln_var)

	def encode_axy_z(self, a, x, y, test=False):
		a = self.to_variable(a)
		x = self.to_variable(x)
		y = self.to_variable(y)
		mean, ln_var = self.q_z_axy(a, x, y, test=test)
		return F.gaussian(mean, ln_var)

	def encode_x_z(self, x, test=False, argmax_y=True):
		x = self.to_variable(x)
		mean, ln_var = self.q_a_x(x, test=test)
		a = F.gaussian(mean, ln_var)
		y = self.sample_x_y(x, argmax=argmax_y, test=test)
		mean, ln_var = self.q_z_axy(a, x, y, test=test)
		return F.gaussian(mean, ln_var)

	def encode_ax_y_distribution(self, a, x, test=False, softmax=True):
		a = self.to_variable(a)
		x = self.to_variable(x)
		unnormalized_distribution = self.q_y_ax(a, x, test=test)
		if softmax:
			return F.softmax(unnormalized_distribution)
		return unnormalized_distribution

	def sample_x_y(self, x, argmax=False, test=False):
		x = self.to_variable(x)
		mean, ln_var = self.q_a_x(x, test=test)
		a = F.gaussian(mean, ln_var)
		return self.sample_ax_y(a, x, argmax=argmax, test=test)

	def sample_ax_y(self, a, x, argmax=False, test=False):
		a = self.to_variable(a)
		x = self.to_variable(x)
		batchsize = self.get_batchsize(x)
		y_distribution = F.softmax(self.q_y_ax(a, x, test=test)).data
		n_labels = y_distribution.shape[1]
		if self.gpu_enabled:
			y_distribution = cuda.to_cpu(y_distribution)
		sampled_y = np.zeros((batchsize, n_labels), dtype=np.float32)
		if argmax:
			args = np.argmax(y_distribution, axis=1)
			for b in xrange(batchsize):
				sampled_y[b, args[b]] = 1
		else:
			for b in xrange(batchsize):
				label_id = np.random.choice(np.arange(n_labels), p=y_distribution[b])
				sampled_y[b, label_id] = 1
		sampled_y = Variable(sampled_y)
		if self.gpu_enabled:
			sampled_y.to_gpu()
		return sampled_y

	def sample_x_y_gumbel(self, x, temperature=10, test=False):
		x = self.to_variable(x)
		mean, ln_var = self.q_a_x(x, test=test)
		a = F.gaussian(mean, ln_var)
		return self.sample_ax_y_gumbel(a, x, temperature=temperature, test=test)

	def sample_ax_y_gumbel(self, a, x, temperature=10, test=False):
		a = self.to_variable(a)
		x = self.to_variable(x)
		batchsize = self.get_batchsize(x)
		log_q_y = self.q_y_ax(a, x, test=test)
		eps = 1e-16
		u = np.random.uniform(0, 1, log_q_y.shape).astype(x.dtype)
		g = self.to_variable(-np.log(-np.log(u + eps) + eps))
		sampled_y = F.softmax((log_q_y + g) / temperature)
		return sampled_y

	def sample_x_label(self, x, argmax=True, test=False):
		x = self.to_variable(x)
		mean, ln_var = self.q_a_x(x, test=test)
		a = F.gaussian(mean, ln_var)
		return self.sample_ax_label(a, x, argmax=argmax, test=test)

	def sample_ax_label(self, a, x, argmax=True, test=False):
		a = self.to_variable(a)
		x = self.to_variable(x)
		batchsize = x.data.shape[0]
		y_distribution = self.q_y_ax(a, x, test=test, softmax=True).data
		n_labels = y_distribution.shape[1]
		if self.gpu_enabled:
			y_distribution = cuda.to_cpu(y_distribution)
		if argmax:
			sampled_label = np.argmax(y_distribution, axis=1)
		else:
			sampled_label = np.zeros((batchsize,), dtype=np.int32)
			labels = np.arange(n_labels)
			for b in xrange(batchsize):
				label_id = np.random.choice(labels, p=y_distribution[b])
				sampled_label[b] = 1
		return sampled_label

	def bernoulli_nll_keepbatch(self, x, y):
		nll = F.softplus(y) - x * y
		return F.sum(nll, axis=1)

	def gaussian_nll_keepbatch(self, x, mean, ln_var, clip=True):
		if clip:
			clip_min = math.log(0.01)
			clip_max = math.log(10)
			ln_var = F.clip(ln_var, clip_min, clip_max)
		x_prec = F.exp(-ln_var)
		x_diff = x - mean
		x_power = (x_diff * x_diff) * x_prec * 0.5
		# print "nll"
		# print cuda.cupy.amax(x.data), cuda.cupy.amin(x.data)
		# print cuda.cupy.amax(ln_var.data), cuda.cupy.amin(ln_var.data)
		# print cuda.cupy.amax(x_prec.data), cuda.cupy.amin(x_prec.data)
		# print cuda.cupy.amax(x_power.data), cuda.cupy.amin(x_power.data)
		return F.sum((math.log(2.0 * math.pi) + ln_var) * 0.5 + x_power, axis=1)

	def gaussian_kl_divergence_keepbatch(self, mean, ln_var):
		var = F.exp(ln_var)
		kld = F.sum(mean * mean + var - ln_var - 1, axis=1) * 0.5
		return kld

	def log_pa(self, a_q, x, y, z, test=False):
		a_mean_p, a_ln_var_p = self.p_a_xyz(x, y, z, test=test)
		negative_log_likelihood = self.gaussian_nll_keepbatch(a_q, a_mean_p, a_ln_var_p)
		log_px_yz = -negative_log_likelihood
		return log_px_yz

	def log_py(self, y):
		xp = self.xp
		n_types_of_label = y.data.shape[1]
		# prior p(y) expecting that all classes are evenly distributed
		constant = math.log(1.0 / n_types_of_label)
		log_py = xp.full((y.data.shape[0],), constant, xp.float32)
		return self.to_variable(log_py)

	def log_pz(self, z):
		log_pz = -0.5 * math.log(2.0 * math.pi) - 0.5 * z ** 2
		return F.sum(log_pz, axis=1)

	# compute lower bound using gumbel-softmax
	def compute_lower_bound_gumbel(self, x_l_cpu_data, y_l_cpu_data, x_u_cpu_data, temperature, test=False):
		assert(isinstance(x_l_cpu_data, np.ndarray))

		def lower_bound(log_px, log_py, log_pa, log_pz, log_qz, log_qa):
			return log_px + log_py + log_pa + log_pz - log_qz - log_qa

		# _l: labeled
		# _u: unlabeled
		batchsize_l = x_l_cpu_data.shape[0]
		batchsize_u = x_u_cpu_data.shape[0]
		n_types_of_label = y_l_cpu_data.shape[1]
		ndim_x = x_l_cpu_data.shape[1]
		num_mc_samples = self.config.num_mc_samples
		xp = self.xp

		### lower bound of labeled data ###
		# repeat num_mc_samples times
		if num_mc_samples == 1:
			x_l = self.to_variable(x_l_cpu_data)
			y_l = self.to_variable(y_l_cpu_data)
		else:
			x_l = self.to_variable(np.repeat(x_l_cpu_data, num_mc_samples, axis=0))
			y_l = self.to_variable(np.repeat(y_l_cpu_data, num_mc_samples, axis=0))

		a_mean_l, a_ln_var_l = self.q_a_x(x_l, test=test)
		a_l = F.gaussian(a_mean_l, a_ln_var_l)
		z_mean_l, z_ln_var_l = self.q_z_axy(a_l, x_l, y_l, test=test)
		z_l = F.gaussian(z_mean_l, z_ln_var_l)

		# compute lower bound
		log_pa_l = self.log_pa(a_l, x_l, y_l, z_l, test=test)
		log_px_l = self.log_px(a_l, x_l, y_l, z_l, test=test)
		log_py_l = self.log_py(y_l)
		log_pz_l = self.log_pz(z_l)
		log_qa_l = -self.gaussian_nll_keepbatch(a_l, a_mean_l, a_ln_var_l)	# 'gaussian_nll_keepbatch' returns the negative log-likelihood
		log_qz_l = -self.gaussian_nll_keepbatch(z_l, z_mean_l, z_ln_var_l)
		lower_bound_l = lower_bound(log_px_l, log_py_l, log_pa_l, log_pz_l, log_qz_l, log_qa_l)

		# take the average
		if num_mc_samples > 1:
			lower_bound_l /= num_mc_samples

		### lower bound of unlabeled data ###
		if batchsize_u > 0:
			# repeat num_mc_samples times
			if num_mc_samples == 1:
				x_u = self.to_variable(x_u_cpu_data)
			else:
				x_u = self.to_variable(np.repeat(x_u_cpu_data, num_mc_samples, axis=0))

			a_mean_u, a_ln_var_u = self.q_a_x(x_u, test=test)
			a_u = F.gaussian(a_mean_u, a_ln_var_u)
			y_u = self.sample_ax_y_gumbel(a_u, x_u, temperature)
			z_mean_u, z_ln_var_u = self.q_z_axy(a_u, x_u, y_u, test=test)
			z_u = F.gaussian(z_mean_u, z_ln_var_u)

			# compute lower bound
			log_pa_u = self.log_pa(a_u, x_u, y_u, z_u, test=test)
			log_px_u = self.log_px(a_u, x_u, y_u, z_u, test=test)
			log_py_u = self.log_py(y_u)
			log_pz_u = self.log_pz(z_u)
			log_qa_u = -self.gaussian_nll_keepbatch(a_u, a_mean_u, a_ln_var_u)	# 'gaussian_nll_keepbatch' returns the negative log-likelihood
			log_qz_u = -self.gaussian_nll_keepbatch(z_u, z_mean_u, z_ln_var_u)
			lower_bound_u = lower_bound(log_px_u, log_py_u, log_pa_u, log_pz_u, log_qz_u, log_qa_u)

			# take the average
			if num_mc_samples > 1:
				lower_bound_u /= num_mc_samples

			lb_labeled = F.sum(lower_bound_l) / batchsize_l
			lb_unlabeled = F.sum(lower_bound_u) / batchsize_u
			lower_bound = lb_labeled + lb_unlabeled
		else:
			lb_unlabeled = None
			lb_labeled = F.sum(lower_bound_l) / batchsize_l
			lower_bound = lb_labeled

		return lower_bound, lb_labeled, lb_unlabeled

	# compute lower bound by marginalizing out y over all classes
	def compute_lower_bound(self, x_l_cpu_data, y_l_cpu_data, x_u_cpu_data, test=False):
		assert(isinstance(x_l_cpu_data, np.ndarray))

		def lower_bound(log_px, log_py, log_pa, log_pz, log_qz, log_qa):
			return log_px + log_py + log_pa + log_pz - log_qz - log_qa


		# _l: labeled
		# _u: unlabeled
		batchsize_l = x_l_cpu_data.shape[0]
		batchsize_u = x_u_cpu_data.shape[0]
		ndim_x = x_u_cpu_data.shape[1]
		n_types_of_label = y_l_cpu_data.shape[1]
		num_mc_samples = self.config.num_mc_samples
		xp = self.xp

		### lower bound of labeled data ###
		# repeat num_mc_samples times
		if num_mc_samples == 1:
			x_l = self.to_variable(x_l_cpu_data)
			y_l = self.to_variable(y_l_cpu_data)
		else:
			x_l = self.to_variable(np.repeat(x_l_cpu_data, num_mc_samples, axis=0))
			y_l = self.to_variable(np.repeat(y_l_cpu_data, num_mc_samples, axis=0))

		a_mean_l, a_ln_var_l = self.q_a_x(x_l, test=test)
		a_l = F.gaussian(a_mean_l, a_ln_var_l)
		z_mean_l, z_ln_var_l = self.q_z_axy(a_l, x_l, y_l, test=test)
		z_l = F.gaussian(z_mean_l, z_ln_var_l)

		# compute lower bound
		log_pa_l = self.log_pa(a_l, x_l, y_l, z_l, test=test)
		log_px_l = self.log_px(a_l, x_l, y_l, z_l, test=test)
		log_py_l = self.log_py(y_l)
		log_pz_l = self.log_pz(z_l)
		log_qa_l = -self.gaussian_nll_keepbatch(a_l, a_mean_l, a_ln_var_l)	# 'gaussian_nll_keepbatch' returns the negative log-likelihood
		log_qz_l = -self.gaussian_nll_keepbatch(z_l, z_mean_l, z_ln_var_l)
		lower_bound_l = lower_bound(log_px_l, log_py_l, log_pa_l, log_pz_l, log_qz_l, log_qa_l)

		# take the average
		if num_mc_samples > 1:
			lower_bound_l /= num_mc_samples

		### lower bound of unlabeled data ###
		if batchsize_u > 0:
			# To marginalize y, we repeat unlabeled x, and construct a target (batchsize_u * n_types_of_label) x n_types_of_label
			# Example of n-dimensional x and target matrix for a 3 class problem and batch_size=2.
			#       x_u              y_repeat
			#  [[x0[0], x0[1], ..., x0[n]]         [[1, 0, 0]
			#   [x1[0], x1[1], ..., x1[n]]          [1, 0, 0]
			#   [x0[0], x0[1], ..., x0[n]]          [0, 1, 0]
			#   [x1[0], x1[1], ..., x1[n]]          [0, 1, 0]
			#   [x0[0], x0[1], ..., x0[n]]          [0, 0, 1]
			#   [x1[0], x1[1], ..., x1[n]]]         [0, 0, 1]]

			# marginalize x and y
			x_u_marg = np.broadcast_to(x_u_cpu_data, (n_types_of_label, batchsize_u, ndim_x)).reshape((batchsize_u * n_types_of_label, ndim_x))
			y_u_marg = np.repeat(np.identity(n_types_of_label, dtype=np.float32), batchsize_u, axis=0)

			# repeat num_mc_samples times
			x_u = x_u_marg
			y_u = y_u_marg
			if num_mc_samples > 1:
				n_rows_marg = x_u_marg.shape[0]
				n_rows = n_rows_marg * num_mc_samples
				x_u = np.repeat(x_u_marg, num_mc_samples, axis=0)
				y_u = np.repeat(y_u_marg, num_mc_samples, axis=0)

			x_u = self.to_variable(x_u)
			y_u = self.to_variable(y_u)

			a_mean_u, a_ln_var_u = self.q_a_x(x_u, test=test)
			a_u = F.gaussian(a_mean_u, a_ln_var_u)
			z_mean_u, z_ln_var_u = self.q_z_axy(a_u, x_u, y_u, test=test)
			z_u = F.gaussian(z_mean_u, z_ln_var_u)

			# compute lower bound
			log_pa_u = self.log_pa(a_u, x_u, y_u, z_u, test=test)
			log_px_u = self.log_px(a_u, x_u, y_u, z_u, test=test)
			log_py_u = self.log_py(y_u)
			log_pz_u = self.log_pz(z_u)
			log_qa_u = -self.gaussian_nll_keepbatch(a_u, a_mean_u, a_ln_var_u)	# 'gaussian_nll_keepbatch' returns the negative log-likelihood
			log_qz_u = -self.gaussian_nll_keepbatch(z_u, z_mean_u, z_ln_var_u)
			lower_bound_u = lower_bound(log_px_u, log_py_u, log_pa_u, log_pz_u, log_qz_u, log_qa_u)

			# Compute sum_y{q(y|x){-L(x,y) + H(q(y|x))}}
			# Let LB(xn, y) be the lower bound for an input image xn and a label y (y = 0, 1, ..., 9).
			# Let bs be the batchsize.
			# 
			# lower_bound_u is a vector and it looks like...
			# [LB(x0,0), LB(x1,0), ..., LB(x_bs,0), LB(x0,1), LB(x1,1), ..., LB(x_bs,1), ..., LB(x0,9), LB(x1,9), ..., LB(x_bs,9)]
			# 
			# After reshaping. (axis 1 corresponds to label, axis 2 corresponds to batch)
			# [[LB(x0,0), LB(x1,0), ..., LB(x_bs,0)],
			#  [LB(x0,1), LB(x1,1), ..., LB(x_bs,1)],
			#                   .
			#                   .
			#                   .
			#  [LB(x0,9), LB(x1,9), ..., LB(x_bs,9)]]
			# 
			# After transposing. (axis 1 corresponds to batch)
			# [[LB(x0,0), LB(x0,1), ..., LB(x0,9)],
			#  [LB(x1,0), LB(x1,1), ..., LB(x1,9)],
			#                   .
			#                   .
			#                   .
			#  [LB(x_bs,0), LB(x_bs,1), ..., LB(x_bs,9)]]
			if num_mc_samples == 1:
				lower_bound_u = F.transpose(F.reshape(lower_bound_u, (n_types_of_label, -1)))
			else:
				lower_bound_u = F.reshape(lower_bound_u, (n_types_of_label, num_mc_samples * batchsize_u))
				lower_bound_u = F.transpose(lower_bound_u)

			# take expectations w.r.t y
			if num_mc_samples == 1:
				x_u = self.to_variable(x_u_cpu_data)
			else:
				x_u = self.to_variable(np.repeat(x_u_cpu_data, num_mc_samples, axis=0))

			a_mean_u, a_ln_var_u = self.q_a_x(x_u, test=test)
			a_u = F.gaussian(a_mean_u, a_ln_var_u)
			y_distribution = F.softmax(self.q_y_ax(a_u, x_u, test=test))

			lower_bound_u = y_distribution * (lower_bound_u - F.log(y_distribution + 1e-6))

			# take the average
			if num_mc_samples > 1:
				lower_bound_u /= num_mc_samples

			lb_labeled = F.sum(lower_bound_l) / batchsize_l
			lb_unlabeled = F.sum(lower_bound_u) / batchsize_u
			lower_bound = lb_labeled + lb_unlabeled
		else:
			lb_unlabeled = None
			lb_labeled = F.sum(lower_bound_l) / batchsize_l
			lower_bound = lb_labeled

		return lower_bound, lb_labeled, lb_unlabeled

class ADGM(DGM):
	def __init__(self, params):
		super(ADGM, self).__init__(params)
		params = self.params
		config = self.config
		self.p_a_xyz = sequential.chain.Chain()
		self.p_a_xyz.add_sequence(sequential.from_dict(params["p_a_xyz"]))
		self.p_a_xyz.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)
		self.p_x_yz = sequential.chain.Chain()
		self.p_x_yz.add_sequence(sequential.from_dict(params["p_x_yz"]))
		self.p_x_yz.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)
		self.q_a_x = sequential.chain.Chain()
		self.q_a_x.add_sequence(sequential.from_dict(params["q_a_x"]))
		self.q_a_x.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)
		self.q_y_ax = sequential.chain.Chain()
		self.q_y_ax.add_sequence(sequential.from_dict(params["q_y_ax"]))
		self.q_y_ax.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)
		self.q_z_axy = sequential.chain.Chain()
		self.q_z_axy.add_sequence(sequential.from_dict(params["q_z_axy"]))
		self.q_z_axy.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)

	def to_gpu(self):
		self.p_a_xyz.to_gpu()
		self.p_x_yz.to_gpu()
		self.q_a_x.to_gpu()
		self.q_y_ax.to_gpu()
		self.q_z_axy.to_gpu()
		self._gpu = True

	def decode_yz_x(self, y, z, test=False):
		y = self.to_variable(y)
		z = self.to_variable(z)
		return F.sigmoid(self.p_x_yz(y, z, test=test))

	def decode_xyz_a(self, x, y, z, test=False):
		x = self.to_variable(x)
		y = self.to_variable(y)
		z = self.to_variable(z)
		mean, ln_var = self.p_a_xyz(x, y, z, test=test)
		return F.gaussian(mean, ln_var)

	def log_px(self, a, x, y, z, test=False):
		x = self.to_variable(x)
		y = self.to_variable(y)
		z = self.to_variable(z)
		# do not apply F.sigmoid to the output of the decoder
		raw_output = self.p_x_yz(y, z, test=test)
		negative_log_likelihood = self.bernoulli_nll_keepbatch(x, raw_output)
		log_px_yz = -negative_log_likelihood
		return log_px_yz

	def backprop(self, loss):
		self.p_x_yz.backprop(loss)
		self.p_a_xyz.backprop(loss)
		self.q_a_x.backprop(loss)
		self.q_y_ax.backprop(loss)
		self.q_z_axy.backprop(loss)

	def backprop_classifier(self, loss):
		self.q_y_ax.backprop(loss)

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		self.p_a_xyz.load(dir + "/p_a_xyz.hdf5")
		self.p_x_yz.load(dir + "/p_x_yz.hdf5")
		self.q_a_x.load(dir + "/q_a_x.hdf5")
		self.q_y_ax.load(dir + "/q_y_ax.hdf5")
		self.q_z_axy.load(dir + "/q_z_axy.hdf5")

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		self.p_a_xyz.save(dir + "/p_a_xyz.hdf5")
		self.p_x_yz.save(dir + "/p_x_yz.hdf5")
		self.q_a_x.save(dir + "/q_a_x.hdf5")
		self.q_y_ax.save(dir + "/q_y_ax.hdf5")
		self.q_z_axy.save(dir + "/q_z_axy.hdf5")

class SDGM(DGM):
	def __init__(self, params):
		super(SDGM, self).__init__(params)
		params = self.params
		config = self.config
		self.p_a_yz = sequential.chain.Chain()
		self.p_a_yz.add_sequence(sequential.from_dict(params["p_a_yz"]))
		self.p_a_yz.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)
		self.p_x_ayz = sequential.chain.Chain()
		self.p_x_ayz.add_sequence(sequential.from_dict(params["p_x_ayz"]))
		self.p_x_ayz.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)
		self.q_a_x = sequential.chain.Chain()
		self.q_a_x.add_sequence(sequential.from_dict(params["q_a_x"]))
		self.q_a_x.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)
		self.q_y_ax = sequential.chain.Chain()
		self.q_y_ax.add_sequence(sequential.from_dict(params["q_y_ax"]))
		self.q_y_ax.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)
		self.q_z_axy = sequential.chain.Chain()
		self.q_z_axy.add_sequence(sequential.from_dict(params["q_z_axy"]))
		self.q_z_axy.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)

	def to_gpu(self):
		self.p_a_yz.to_gpu()
		self.p_x_ayz.to_gpu()
		self.q_a_x.to_gpu()
		self.q_y_ax.to_gpu()
		self.q_z_axy.to_gpu()
		self._gpu = True

	def decode_ayz_x(self, a, y, z, test=False):
		a = self.to_variable(a)
		y = self.to_variable(y)
		z = self.to_variable(z)
		return F.sigmoid(self.p_x_ayz(a, y, z, test=test))

	def decode_yz_a(self, y, z, test=False):
		y = self.to_variable(y)
		z = self.to_variable(z)
		mean, ln_var = self.p_a_yz(y, z, test=test)
		return F.gaussian(mean, ln_var)

	def log_px(self, a, x, y, z, test=False):
		a = self.to_variable(a)
		x = self.to_variable(x)
		y = self.to_variable(y)
		z = self.to_variable(z)
		# do not apply F.sigmoid to the output of the decoder
		raw_output = self.p_x_ayz(a, y, z, test=test)
		negative_log_likelihood = self.bernoulli_nll_keepbatch(x, raw_output)
		log_px_yz = -negative_log_likelihood
		return log_px_yz

	def log_pa(self, a_q, x, y, z, test=False):
		a_mean_p, a_ln_var_p = self.p_a_yz(y, z, test=test)
		negative_log_likelihood = self.gaussian_nll_keepbatch(a_q, a_mean_p, a_ln_var_p)
		log_px_yz = -negative_log_likelihood
		return log_px_yz

	def backprop(self, loss):
		self.p_x_ayz.backprop(loss)
		self.p_a_yz.backprop(loss)
		self.q_a_x.backprop(loss)
		self.q_y_ax.backprop(loss)
		self.q_z_axy.backprop(loss)

	def backprop_classifier(self, loss):
		self.q_y_ax.backprop(loss)

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		self.p_a_yz.load(dir + "/p_a_yz.hdf5")
		self.p_x_ayz.load(dir + "/p_x_ayz.hdf5")
		self.q_a_x.load(dir + "/q_a_x.hdf5")
		self.q_y_ax.load(dir + "/q_y_ax.hdf5")
		self.q_z_axy.load(dir + "/q_z_axy.hdf5")

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		self.p_a_yz.save(dir + "/p_a_yz.hdf5")
		self.p_x_ayz.save(dir + "/p_x_ayz.hdf5")
		self.q_a_x.save(dir + "/q_a_x.hdf5")
		self.q_y_ax.save(dir + "/q_y_ax.hdf5")
		self.q_z_axy.save(dir + "/q_z_axy.hdf5")