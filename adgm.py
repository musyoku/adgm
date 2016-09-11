# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six
from chainer import cuda, Variable, optimizers, serializers, function, optimizer
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L

activations = {
	"sigmoid": F.sigmoid, 
	"tanh": F.tanh, 
	"softplus": F.softplus, 
	"relu": F.relu, 
	"leaky_relu": F.leaky_relu, 
	"elu": F.elu
}

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

		self.decoder_yz_x_hidden_units = [500, 500]
		self.decoder_yz_x_activation_function = "elu"
		self.decoder_yz_x_apply_dropout = False
		self.decoder_yz_x_apply_batchnorm = True
		self.decoder_yz_x_apply_batchnorm_to_input = True

		self.decoder_xyz_a_hidden_units = [500, 500]
		self.decoder_xyz_a_activation_function = "elu"
		self.decoder_xyz_a_apply_dropout = False
		self.decoder_xyz_a_apply_batchnorm = True
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

def sum_sqnorm(arr):
	sq_sum = collections.defaultdict(float)
	for x in arr:
		with cuda.get_device(x) as dev:
			x = x.ravel()
			s = x.dot(x)
			sq_sum[int(dev)] += s
	return sum([float(i) for i in six.itervalues(sq_sum)])
	
class GradientClipping(object):
	name = "GradientClipping"

	def __init__(self, threshold):
		self.threshold = threshold

	def __call__(self, opt):
		norm = np.sqrt(sum_sqnorm([p.grad for p in opt.target.params()]))
		if norm == 0:
			return
		rate = self.threshold / norm
		if rate < 1:
			for param in opt.target.params():
				grad = param.grad
				with cuda.get_device(grad):
					grad *= rate

class DGM():

	def build_encoder_axy_z(self):
		conf = self.conf
		attributes = {}
		units = zip(conf.encoder_xy_z_hidden_units[:-1], conf.encoder_xy_z_hidden_units[1:])
		for i, (n_in, n_out) in enumerate(units):
			attributes["layer_%i" % i] = L.Linear(n_in, n_out, initialW=np.random.normal(scale=conf.wscale, size=(n_out, n_in)))
			if conf.batchnorm_before_activation:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		attributes["layer_merge_a"] = L.Linear(conf.ndim_a, conf.encoder_xy_z_hidden_units[0], initialW=np.random.normal(scale=conf.wscale, size=(conf.encoder_xy_z_hidden_units[0], conf.ndim_a)))
		attributes["layer_merge_x"] = L.Linear(conf.ndim_x, conf.encoder_xy_z_hidden_units[0], initialW=np.random.normal(scale=conf.wscale, size=(conf.encoder_xy_z_hidden_units[0], conf.ndim_x)))
		attributes["layer_merge_y"] = L.Linear(conf.ndim_y, conf.encoder_xy_z_hidden_units[0], initialW=np.random.normal(scale=conf.wscale, size=(conf.encoder_xy_z_hidden_units[0], conf.ndim_y)))

		if conf.batchnorm_before_activation:
			attributes["batchnorm_merge"] = L.BatchNormalization(conf.encoder_xy_z_hidden_units[0])
		else:
			attributes["batchnorm_merge_x"] = L.BatchNormalization(conf.ndim_x)
			attributes["batchnorm_merge_a"] = L.BatchNormalization(conf.ndim_a)

		attributes["layer_output_mean"] = L.Linear(conf.encoder_xy_z_hidden_units[-1], conf.ndim_z, initialW=np.random.normal(scale=conf.wscale, size=(conf.ndim_z, conf.encoder_xy_z_hidden_units[-1])))
		attributes["layer_output_var"] = L.Linear(conf.encoder_xy_z_hidden_units[-1], conf.ndim_z, initialW=np.random.normal(scale=conf.wscale, size=(conf.ndim_z, conf.encoder_xy_z_hidden_units[-1])))
		encoder_axy_z = GaussianEncoder_AXY_Z(**attributes)
		encoder_axy_z.n_layers = len(units)
		encoder_axy_z.activation_function = conf.encoder_xy_z_activation_function
		encoder_axy_z.apply_dropout = conf.encoder_xy_z_apply_dropout
		encoder_axy_z.apply_batchnorm = conf.encoder_xy_z_apply_batchnorm
		encoder_axy_z.apply_batchnorm_to_input = conf.encoder_xy_z_apply_batchnorm_to_input
		encoder_axy_z.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			encoder_axy_z.to_gpu()

		return encoder_axy_z

	def build_encoder_ax_y(self):
		conf = self.conf
		attributes = {}
		units = zip(conf.encoder_ax_y_hidden_units[:-1], conf.encoder_ax_y_hidden_units[1:])
		units += [(conf.encoder_ax_y_hidden_units[-1], conf.ndim_y)]
		for i, (n_in, n_out) in enumerate(units):
			attributes["layer_%i" % i] = L.Linear(n_in, n_out, initialW=np.random.normal(scale=conf.wscale, size=(n_out, n_in)))
			if conf.batchnorm_before_activation:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		attributes["layer_merge_x"] = L.Linear(conf.ndim_x, conf.encoder_ax_y_hidden_units[0], initialW=np.random.normal(scale=conf.wscale, size=(conf.encoder_ax_y_hidden_units[0], conf.ndim_x)))
		attributes["layer_merge_a"] = L.Linear(conf.ndim_a, conf.encoder_ax_y_hidden_units[0], initialW=np.random.normal(scale=conf.wscale, size=(conf.encoder_ax_y_hidden_units[0], conf.ndim_a)))
		attributes["batchnorm_merge"] = L.BatchNormalization(conf.encoder_ax_y_hidden_units[0])
		encoder_ax_y = SoftmaxEncoder_AX_Y(**attributes)
		encoder_ax_y.n_layers = len(units)
		encoder_ax_y.activation_function = conf.encoder_ax_y_activation_function
		encoder_ax_y.apply_dropout = conf.encoder_ax_y_apply_dropout
		encoder_ax_y.apply_batchnorm = conf.encoder_ax_y_apply_batchnorm
		encoder_ax_y.apply_batchnorm_to_input = conf.encoder_ax_y_apply_batchnorm_to_input
		encoder_ax_y.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			encoder_ax_y.to_gpu()

		return encoder_ax_y

	def build_encoder_x_a(self):
		conf = self.conf
		attributes = {}
		units = [(conf.ndim_x, conf.encoder_x_a_hidden_units[0])]
		units += zip(conf.encoder_x_a_hidden_units[:-1], conf.encoder_x_a_hidden_units[1:])
		for i, (n_in, n_out) in enumerate(units):
			attributes["layer_%i" % i] = L.Linear(n_in, n_out, initialW=np.random.normal(scale=conf.wscale, size=(n_out, n_in)))
			if conf.batchnorm_before_activation:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		attributes["layer_output_mean"] = L.Linear(conf.encoder_x_a_hidden_units[-1], conf.ndim_a, initialW=np.random.normal(scale=conf.wscale, size=(conf.ndim_a, conf.encoder_x_a_hidden_units[-1])))
		attributes["layer_output_var"] = L.Linear(conf.encoder_x_a_hidden_units[-1], conf.ndim_a, initialW=np.random.normal(scale=conf.wscale, size=(conf.ndim_a, conf.encoder_x_a_hidden_units[-1])))
		encoder_x_a = GaussianEncoder_X_A(**attributes)
		encoder_x_a.n_layers = len(units)
		encoder_x_a.activation_function = conf.encoder_x_a_activation_function
		encoder_x_a.apply_dropout = conf.encoder_x_a_apply_dropout
		encoder_x_a.apply_batchnorm = conf.encoder_x_a_apply_batchnorm
		encoder_x_a.apply_batchnorm_to_input = conf.encoder_x_a_apply_batchnorm_to_input
		encoder_x_a.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			encoder_x_a.to_gpu()

		return encoder_x_a

	def build_decoder_xyz_a(self):
		conf = self.conf
		attributes = {}
		units = zip(conf.decoder_xyz_a_hidden_units[:-1], conf.decoder_xyz_a_hidden_units[1:])
		for i, (n_in, n_out) in enumerate(units):
			attributes["layer_%i" % i] = L.Linear(n_in, n_out, initialW=np.random.normal(scale=conf.wscale, size=(n_out, n_in)))
			if conf.batchnorm_before_activation:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		attributes["layer_merge_x"] = L.Linear(conf.ndim_x, conf.decoder_xyz_a_hidden_units[0], initialW=np.random.normal(scale=conf.wscale, size=(conf.decoder_xyz_a_hidden_units[0], conf.ndim_x)))
		attributes["layer_merge_y"] = L.Linear(conf.ndim_y, conf.decoder_xyz_a_hidden_units[0], initialW=np.random.normal(scale=conf.wscale, size=(conf.decoder_xyz_a_hidden_units[0], conf.ndim_y)))
		attributes["layer_merge_z"] = L.Linear(conf.ndim_z, conf.decoder_xyz_a_hidden_units[0], initialW=np.random.normal(scale=conf.wscale, size=(conf.decoder_xyz_a_hidden_units[0], conf.ndim_z)))

		if conf.batchnorm_before_activation:
			attributes["batchnorm_merge"] = L.BatchNormalization(conf.decoder_xyz_a_hidden_units[0])
		else:
			attributes["batchnorm_merge_x"] = L.BatchNormalization(conf.ndim_x)
			attributes["batchnorm_merge_z"] = L.BatchNormalization(conf.ndim_z)

		attributes["layer_output_mean"] = L.Linear(conf.decoder_xyz_a_hidden_units[-1], conf.ndim_a, initialW=np.random.normal(scale=conf.wscale, size=(conf.ndim_a, conf.decoder_xyz_a_hidden_units[-1])))
		attributes["layer_output_var"] = L.Linear(conf.decoder_xyz_a_hidden_units[-1], conf.ndim_a, initialW=np.random.normal(scale=conf.wscale, size=(conf.ndim_a, conf.decoder_xyz_a_hidden_units[-1])))
		decoder_xyz_a = GaussianDecoder_XYZ_A(**attributes)
		decoder_xyz_a.n_layers = len(units)
		decoder_xyz_a.activation_function = conf.decoder_xyz_a_activation_function
		decoder_xyz_a.apply_dropout = conf.decoder_xyz_a_apply_dropout
		decoder_xyz_a.apply_batchnorm = conf.decoder_xyz_a_apply_batchnorm
		decoder_xyz_a.apply_batchnorm_to_input = conf.decoder_xyz_a_apply_batchnorm_to_input
		decoder_xyz_a.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			decoder_xyz_a.to_gpu()

		return decoder_xyz_a

	@property
	def xp(self):
		return self.encoder_axy_z.xp

	@property
	def gpu(self):
		if cuda.available is False:
			return False
		return True if self.xp is cuda.cupy else False

	def update_classifier(self):
		self.optimizer_encoder_ax_y.update()

	def encode_x_a(self, x, test=False, apply_f=True):
		return self.encoder_x_a(x, test=test, apply_f=apply_f)

	def encode_axy_z(self, a, x, y, test=False, apply_f=True):
		return self.encoder_axy_z(a, x, y, test=test, apply_f=apply_f)

	def encode_x_z(self, x, test=False, argmax=True):
		a = self.encoder_x_a(x, test=test, apply_f=True)
		y = self.sample_x_y(x, argmax=argmax, test=test)
		return self.encoder_axy_z(a, x, y, test=test, apply_f=True)

	def decode_xyz_a(self, x, y, z, test=False, apply_f=True):
		return self.decoder_xyz_a(x, y, z, test=test, apply_f=apply_f)

	def sample_x_y(self, x, argmax=False, test=False):
		a = self.encoder_x_a(x, test=test, apply_f=True)
		return self.sample_ax_y(a, x, argmax=argmax, test=test)

	def sample_ax_y(self, a, x, argmax=False, test=False):
		batchsize = x.data.shape[0]
		y_distribution = self.encoder_ax_y(a, x, test=test, softmax=True).data
		n_labels = y_distribution.shape[1]
		if self.gpu:
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
		if self.gpu:
			sampled_y.to_gpu()
		return sampled_y

	def sample_x_label(self, x, argmax=True, test=False):
		a = self.encoder_x_a(x, test=test, apply_f=True)
		return self.sample_ax_label(a, x, argmax=argmax, test=test)

	def sample_ax_label(self, a, x, argmax=True, test=False):
		batchsize = x.data.shape[0]
		a = self.encoder_x_a(x, test=test, apply_f=True)
		y_distribution = self.encoder_ax_y(a, x, test=test, softmax=True).data
		n_labels = y_distribution.shape[1]
		if self.gpu:
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
		return F.sum((math.log(2.0 * math.pi) + ln_var) * 0.5 + x_power, axis=1)

	def gaussian_kl_divergence_keepbatch(self, mean, ln_var):
		var = F.exp(ln_var)
		kld = F.sum(mean * mean + var - ln_var - 1, axis=1) * 0.5
		return kld

	def log_pa(self, a, x, y, z, test=False):
		a_mean, a_ln_var = self.decoder_xyz_a(x, y, z, test=test, apply_f=False)
		negative_log_likelihood = self.gaussian_nll_keepbatch(a, a_mean, a_ln_var)
		log_px_yz = -negative_log_likelihood
		return log_px_yz

	def log_py(self, y):
		xp = self.xp
		n_types_of_label = y.data.shape[1]
		# prior p(y) expecting that all classes are evenly distributed
		constant = math.log(1.0 / n_types_of_label)
		log_py = xp.full((y.data.shape[0],), constant, xp.float32)
		return Variable(log_py)

	def log_pz(self, z):
		log_pz = -0.5 * math.log(2.0 * math.pi) - 0.5 * z ** 2
		return F.sum(log_pz, axis=1)

	def train(self, labeled_x, labeled_y, label_ids, unlabeled_x):
		loss, loss_labeled, loss_unlabeled = self.compute_lower_bound_loss(labeled_x, labeled_y, label_ids, unlabeled_x, test=False)

		self.zero_grads()
		loss.backward()
		self.update()

		if self.gpu:
			loss_labeled.to_cpu()
			if loss_unlabeled is not None:
				loss_unlabeled.to_cpu()

		if loss_unlabeled is None:
			return loss_labeled.data, 0

		return loss_labeled.data, loss_unlabeled.data

	def train_classification(self, labeled_x, label_ids, alpha=1.0):
		loss = alpha * self.compute_classification_loss(labeled_x, label_ids, test=False)
		self.zero_grads()
		loss.backward()
		self.update_classifier()
		if self.gpu:
			loss.to_cpu()
		return loss.data

	def train_jointly(self, labeled_x, labeled_y, label_ids, unlabeled_x, alpha=1.0):
		loss_lower_bound, loss_lb_labled, loss_lb_unlabled = self.compute_lower_bound_loss(labeled_x, labeled_y, label_ids, unlabeled_x, test=False)

		loss_classification = alpha * self.compute_classification_loss(labeled_x, label_ids, test=False)
		loss = loss_lower_bound + loss_classification
		self.zero_grads()
		loss.backward()
		self.update()
		if self.gpu:
			loss_lb_labled.to_cpu()
			if loss_lb_unlabled is not None:
				loss_lb_unlabled.to_cpu()
			loss_classification.to_cpu()

		if loss_lb_unlabled is None:
			return loss_lb_labled.data, 0, loss_classification.data

		return loss_lb_labled.data, loss_lb_unlabled.data, loss_classification.data

	def compute_classification_loss(self, labeled_x, label_ids, test=False):
		a = self.encoder_x_a(labeled_x, test=test, apply_f=True)
		y_distribution = self.encoder_ax_y(a, labeled_x, softmax=False, test=test)
		batchsize = labeled_x.data.shape[0]
		n_types_of_label = y_distribution.data.shape[1]

		loss = F.softmax_cross_entropy(y_distribution, label_ids)
		return loss

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod):
				filename = dir + "/%s_%s.hdf5" % (self.name, attr)
				if os.path.isfile(filename):
					print "loading",  filename
					serializers.load_hdf5(filename, prop)
				else:
					print filename, "not found."
		print "model loaded."

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod):
				serializers.save_hdf5(dir + "/%s_%s.hdf5" % (self.name, attr), prop)
		print "model saved."


	def compute_lower_bound_loss(self, labeled_x, labeled_y, label_ids, unlabeled_x, test=False):

		def lower_bound(log_px, log_py, log_pa, log_pz, log_qz, log_qa):
			return log_px + log_py + log_pa + log_pz - log_qz - log_qa

		# _l: labeled
		# _u: unlabeled
		batchsize_l = labeled_x.data.shape[0]
		batchsize_u = unlabeled_x.data.shape[0]
		n_types_of_label = labeled_y.data.shape[1]
		n_mc_samples = self.conf.n_mc_samples
		xp = self.xp

		### Lower bound for labeled data ###
		labeled_x_repeat = labeled_x
		labeled_y_repeat = labeled_y

		# repeat n_mc_samples times
		if n_mc_samples > 1:
			labeled_x_repeat = Variable(xp.repeat(labeled_x.data, n_mc_samples, axis=0))
			labeled_y_repeat = Variable(xp.repeat(labeled_y.data, n_mc_samples, axis=0))


		a_mean_l, a_ln_var_l = self.encoder_x_a(labeled_x_repeat, test=test, apply_f=False)
		a_l = F.gaussian(a_mean_l, a_ln_var_l)
		z_mean_l, z_ln_var_l = self.encoder_axy_z(a_l, labeled_x_repeat, labeled_y_repeat, test=test, apply_f=False)
		z_l = F.gaussian(z_mean_l, z_ln_var_l)

		# compute lower bound
		log_pa_l = self.log_pa(a_l, labeled_x_repeat, labeled_y_repeat, z_l)
		log_px_l = self.log_px(a_l, labeled_x_repeat, labeled_y_repeat, z_l, test=test)
		log_py_l = self.log_py(labeled_y_repeat)
		log_pz_l = self.log_pz(z_l)
		log_qa_l = -self.gaussian_nll_keepbatch(a_l, a_mean_l, a_ln_var_l)	# 'gaussian_nll_keepbatch' returns the negative log-likelihood
		log_qz_l = -self.gaussian_nll_keepbatch(z_l, z_mean_l, z_ln_var_l)
		lower_bound_l = lower_bound(log_px_l, log_py_l, log_pa_l, log_pz_l, log_qz_l, log_qa_l)

		# take the average
		if n_mc_samples > 1:
			lower_bound_l /= n_mc_samples

		### Lower bound for unlabeled data ###
		if batchsize_u > 0:
			# To marginalize y, we repeat unlabeled x, and construct a target (batchsize_u * n_types_of_label) x n_types_of_label
			# Example of n-dimensional x and target matrix for a 3 class problem and batch_size=2.
			#       unlabeled_x_repeat              y_repeat
			#  [[x0[0], x0[1], ..., x0[n]]         [[1, 0, 0]
			#   [x1[0], x1[1], ..., x1[n]]          [1, 0, 0]
			#   [x0[0], x0[1], ..., x0[n]]          [0, 1, 0]
			#   [x1[0], x1[1], ..., x1[n]]          [0, 1, 0]
			#   [x0[0], x0[1], ..., x0[n]]          [0, 0, 1]
			#   [x1[0], x1[1], ..., x1[n]]]         [0, 0, 1]]

			# marginalize x and y
			unlabeled_x_marg = xp.empty((batchsize_u * n_types_of_label, unlabeled_x.data.shape[1]), dtype=xp.float32)
			y_marg = xp.repeat(xp.identity(n_types_of_label, dtype=xp.float32), batchsize_u, axis=0)
			for n in xrange(n_types_of_label):
				unlabeled_x_marg[n * batchsize_u:(n + 1) * batchsize_u] = unlabeled_x.data

			# repeat n_mc_samples times
			unlabeled_x_repeat = unlabeled_x_marg
			y_repeat = y_marg
			if n_mc_samples > 1:
				n_rows_marg = unlabeled_x_marg.shape[0]
				n_rows = n_rows_marg * n_mc_samples
				unlabeled_x_repeat = xp.repeat(unlabeled_x_marg, n_mc_samples, axis=0)
				y_repeat = xp.repeat(y_marg, n_mc_samples, axis=0)
			unlabeled_x_repeat = Variable(unlabeled_x_repeat)
			y_repeat = Variable(y_repeat)

			a_mean_u, a_ln_var_u = self.encoder_x_a(unlabeled_x_repeat, test=test, apply_f=False)
			a_u = F.gaussian(a_mean_u, a_ln_var_u)
			z_mean_u, z_mean_ln_var_u = self.encoder_axy_z(a_u, unlabeled_x_repeat, y_repeat, test=test, apply_f=False)
			z_u = F.gaussian(z_mean_u, z_mean_ln_var_u)

			# compute lower bound
			log_pa_u = self.log_pa(a_u, unlabeled_x_repeat, y_repeat, z_u)
			log_px_u = self.log_px(a_u, unlabeled_x_repeat, y_repeat, z_u, test=test)
			log_py_u = self.log_py(y_repeat)
			log_pz_u = self.log_pz(z_u)
			log_qa_u = -self.gaussian_nll_keepbatch(a_u, a_mean_u, a_ln_var_u)	# 'gaussian_nll_keepbatch' returns the negative log-likelihood
			log_qz_u = -self.gaussian_nll_keepbatch(z_u, z_mean_u, z_mean_ln_var_u)
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
			if n_mc_samples > 1:
				lower_bound_u = F.reshape(lower_bound_u, (n_types_of_label, n_mc_samples * batchsize_u))
				lower_bound_u = F.transpose(lower_bound_u)
			else:
				lower_bound_u = F.transpose(F.reshape(lower_bound_u, (n_types_of_label, -1)))

			# take expectations w.r.t 'y'
			unlabeled_x_repeat = unlabeled_x
			if n_mc_samples > 1:
				unlabeled_x_repeat = Variable(xp.repeat(unlabeled_x.data, n_mc_samples, axis=0))

			a_u = self.encoder_x_a(unlabeled_x_repeat, test=test, apply_f=True)
			y_distribution = self.encoder_ax_y(a_u, unlabeled_x_repeat, test=test, softmax=True)
			lower_bound_u = y_distribution * (lower_bound_u - F.log(y_distribution + 1e-6))

			# take the average
			if n_mc_samples > 1:
				lower_bound_u /= n_mc_samples

			# loss = -1 * lower bound
			loss_labeled = -F.sum(lower_bound_l) / batchsize_l
			loss_unlabeled = -F.sum(lower_bound_u) / batchsize_u
			loss = loss_labeled + loss_unlabeled
		else:
			loss_unlabeled = None
			loss_labeled = -F.sum(lower_bound_l) / batchsize_l
			loss = loss_labeled

		return loss, loss_labeled, loss_unlabeled

class ADGM(DGM):
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
		self.decoder_yz_x = self.build_decoder_yz_x()
		self.optimizer_decoder_yz_x = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_decoder_yz_x.setup(self.decoder_yz_x)
		# self.optimizer_decoder_yz_x.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_decoder_yz_x.add_hook(GradientClipping(conf.gradient_clipping))

		# p(a|x, y, z)
		self.decoder_xyz_a = self.build_decoder_xyz_a()
		self.optimizer_decoder_xyz_a = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_decoder_xyz_a.setup(self.decoder_xyz_a)
		# self.optimizer_decoder_xyz_a.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_decoder_xyz_a.add_hook(GradientClipping(conf.gradient_clipping))

	def build_decoder_yz_x(self):
		conf = self.conf
		attributes = {}
		units = zip(conf.decoder_yz_x_hidden_units[:-1], conf.decoder_yz_x_hidden_units[1:])
		units += [(conf.decoder_yz_x_hidden_units[-1], conf.ndim_x)]
		for i, (n_in, n_out) in enumerate(units):
			attributes["layer_%i" % i] = L.Linear(n_in, n_out, initialW=np.random.normal(scale=conf.wscale, size=(n_out, n_in)))
			if conf.batchnorm_before_activation:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		attributes["layer_merge_z"] = L.Linear(conf.ndim_z, conf.decoder_yz_x_hidden_units[0], initialW=np.random.normal(scale=conf.wscale, size=(conf.decoder_yz_x_hidden_units[0], conf.ndim_z)))
		attributes["layer_merge_y"] = L.Linear(conf.ndim_y, conf.decoder_yz_x_hidden_units[0], initialW=np.random.normal(scale=conf.wscale, size=(conf.decoder_yz_x_hidden_units[0], conf.ndim_y)))
		attributes["batchnorm_merge"] = L.BatchNormalization(conf.decoder_yz_x_hidden_units[0])

		if conf.distribution_x == "bernoulli":
			decoder_yz_x = BernoulliDecoder_YZ_X(**attributes)
		else:
			attributes["layer_output_mean"] = L.Linear(conf.decoder_yz_x_hidden_units[-1], conf.ndim_x, initialW=np.random.normal(scale=conf.wscale, size=(conf.ndim_x, conf.decoder_yz_x_hidden_units[-1])))
			attributes["layer_output_var"] = L.Linear(conf.decoder_yz_x_hidden_units[-1], conf.ndim_x, initialW=np.random.normal(scale=conf.wscale, size=(conf.ndim_x, conf.decoder_yz_x_hidden_units[-1])))
			decoder_yz_x = GaussianDecoder_YZ_X(**attributes)
		decoder_yz_x.n_layers = len(units)
		decoder_yz_x.activation_function = conf.decoder_yz_x_activation_function
		decoder_yz_x.apply_dropout = conf.decoder_yz_x_apply_dropout
		decoder_yz_x.apply_batchnorm = conf.decoder_yz_x_apply_batchnorm
		decoder_yz_x.apply_batchnorm_to_input = conf.decoder_yz_x_apply_batchnorm_to_input
		decoder_yz_x.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			decoder_yz_x.to_gpu()

		return decoder_yz_x

	def zero_grads(self):
		self.optimizer_encoder_axy_z.zero_grads()
		self.optimizer_encoder_ax_y.zero_grads()
		self.optimizer_encoder_x_a.zero_grads()
		self.optimizer_decoder_yz_x.zero_grads()
		self.optimizer_decoder_xyz_a.zero_grads()

	def update(self):
		self.optimizer_encoder_axy_z.update()
		self.optimizer_encoder_ax_y.update()
		self.optimizer_encoder_x_a.update()
		self.optimizer_decoder_yz_x.update()
		self.optimizer_decoder_xyz_a.update()

	def decode_yz_x(self, y, z, test=False, apply_f=True):
		return self.decoder_yz_x(y, z, test=test, apply_f=apply_f)

	def log_px(self, a, x, y, z, test=False):
		if isinstance(self.decoder_yz_x, BernoulliDecoder_YZ_X):
			# do not apply F.sigmoid to the output of the decoder
			raw_output = self.decoder_yz_x(y, z, test=test, apply_f=False)
			negative_log_likelihood = self.bernoulli_nll_keepbatch(x, raw_output)
			log_px_yz = -negative_log_likelihood
		else:
			x_mean, x_ln_var = self.decoder_yz_x(y, z, test=test, apply_f=False)
			negative_log_likelihood = self.gaussian_nll_keepbatch(x, x_mean, x_ln_var)
			log_px_yz = -negative_log_likelihood
		return log_px_yz

class MultiLayerPerceptron(chainer.Chain):
	def __init__(self, **layers):
		super(MultiLayerPerceptron, self).__init__(**layers)
		self.activation_function = "elu"
		self.apply_batchnorm_to_input = True
		self.apply_batchnorm = True
		self.apply_dropout = False
		self.batchnorm_before_activation = True
		self.test = False

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def compute_output(self, x):
		f = activations[self.activation_function]
		chain = [x]

		# Hidden
		for i in range(self.n_layers):
			u = chain[-1]
			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)
			if i == self.n_layers - 1:
				if self.apply_batchnorm and self.batchnorm_before_activation == False:
					u = getattr(self, "batchnorm_%d" % i)(u, test=self.test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "batchnorm_%d" % i)(u, test=self.test)
			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)
			if i == self.n_layers - 1:
				output = u
			else:
				output = f(u)
				if self.apply_dropout:
					output = F.dropout(output, train=not self.test)
			chain.append(output)

		return chain[-1]

	def __call__(self, x, test=False, apply_f=True):
		self.test = test
		output = self.compute_output(x)
		if apply_f:
			f = activations[self.activation_function]
			return f(output)
		return output

class SoftmaxEncoder_AX_Y(MultiLayerPerceptron):

	def forward_one_step(self, a, x):
		f = activations[self.activation_function]
		if self.apply_batchnorm_to_input:
			if self.batchnorm_before_activation:
				merged_input = f(self.batchnorm_merge(self.layer_merge_a(a) + self.layer_merge_x(x), test=self.test))
			else:
				merged_input = f(self.layer_merge_a(self.batchnorm_merge(a, test=self.test)) + self.layer_merge_x(x))
		else:
			merged_input = f(self.layer_merge_a(a) + self.layer_merge_x(x))
		return self.compute_output(merged_input)

	def __call__(self, a, x, test=False, softmax=False):
		self.test = test
		output = self.forward_one_step(a, x)
		if softmax:
			return F.softmax(output)
		return output

class GaussianEncoder(chainer.Chain):
	def __init__(self, **layers):
		super(GaussianEncoder, self).__init__(**layers)
		self.activation_function = "softplus"
		self.apply_batchnorm_to_input = True
		self.apply_batchnorm = True
		self.apply_dropout = False
		self.batchnorm_before_activation = True

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def forward_one_step(self, x):
		return self.compute_output(x)

	def compute_output(self, x):
		f = activations[self.activation_function]
		chain = [x]
		# Hidden
		for i in range(self.n_layers):
			u = chain[-1]
			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)
			if self.apply_batchnorm:
				u = getattr(self, "batchnorm_%d" % i)(u, test=self.test)
			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)
			output = f(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not self.test)
			chain.append(output)

		u = chain[-1]
		mean = self.layer_output_mean(u)

		# log(sd^2)
		u = chain[-1]
		ln_var = self.layer_output_var(u)

		return mean, ln_var

	def __call__(self, x, test=False, apply_f=True):
		self.test = test
		mean, ln_var = self.forward_one_step(x)
		if apply_f:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

class GaussianEncoder_X_A(GaussianEncoder):
	pass

class GaussianEncoder_AXY_Z(GaussianEncoder):

	def merge_input(self, a, x, y):
		f = activations[self.activation_function]
		if self.apply_batchnorm_to_input:
			if self.batchnorm_before_activation:
				merged_input = f(self.batchnorm_merge(self.layer_merge_a(a) + self.layer_merge_y(y) + self.layer_merge_x(x), test=self.test))
			else:
				merged_input = f(self.layer_merge_a(self.batchnorm_merge_z(a, test=self.test)) + self.layer_merge_y(y) + self.layer_merge_x(self.batchnorm_merge_x(x, test=self.test)))
		else:
			merged_input = f(self.layer_merge_a(a) + self.layer_merge_y(y) + self.layer_merge_x(x))

		return merged_input

	def forward_one_step(self, a, x, y):
		merged_input = self.merge_input(a, x, y)
		return self.compute_output(merged_input)

	def __call__(self, a, x, y, test=False, apply_f=False):
		self.test = test
		mean, ln_var = self.forward_one_step(a, x, y)
		if apply_f:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

class GaussianDecoder_XYZ_A(GaussianEncoder):

	def merge_input(self, x, y, z):
		f = activations[self.activation_function]
		if self.apply_batchnorm_to_input:
			if self.batchnorm_before_activation:
				merged_input = f(self.batchnorm_merge(self.layer_merge_z(z) + self.layer_merge_y(y) + self.layer_merge_x(x), test=self.test))
			else:
				merged_input = f(self.layer_merge_z(self.batchnorm_merge_z(z, test=self.test)) + self.layer_merge_y(y) + self.layer_merge_x(self.batchnorm_merge_x(x, test=self.test)))
		else:
			merged_input = f(self.layer_merge_z(z) + self.layer_merge_y(y) + self.layer_merge_x(x))

		return merged_input

	def forward_one_step(self, x, y, z):
		merged_input = self.merge_input(x, y, z)
		return self.compute_output(merged_input)

	def __call__(self, x, y, z, test=False, apply_f=False):
		self.test = test
		mean, ln_var = self.forward_one_step(x, y, z)
		if apply_f:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

class GaussianDecoder_YZ_X(GaussianEncoder):

	def merge_input(self, y, z):
		f = activations[self.activation_function]
		if self.apply_batchnorm_to_input:
			if self.batchnorm_before_activation:
				merged_input = f(self.batchnorm_merge(self.layer_merge_z(z) + self.layer_merge_y(y), test=self.test))
			else:
				merged_input = f(self.layer_merge_z(self.batchnorm_merge(z, test=self.test)) + self.layer_merge_y(y))
		else:
			merged_input = f(self.layer_merge_z(z) + self.layer_merge_y(y))

		return merged_input

	def forward_one_step(self, y, z):
		merged_input = self.merge_input(y, z)
		return self.compute_output(merged_input)

	def __call__(self, y, z, test=False, apply_f=False):
		self.test = test
		mean, ln_var = self.forward_one_step(y, z)
		if apply_f:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

class BernoulliDecoder_YZ_X(MultiLayerPerceptron):

	def forward_one_step(self, y, z):
		f = activations[self.activation_function]
		if self.apply_batchnorm_to_input:
			if self.batchnorm_before_activation:
				merged_input = f(self.batchnorm_merge(self.layer_merge_z(z) + self.layer_merge_y(y), test=self.test))
			else:
				merged_input = f(self.layer_merge_z(self.batchnorm_merge(z, test=self.test)) + self.layer_merge_y(y))
		else:
			merged_input = f(self.layer_merge_z(z) + self.layer_merge_y(y))
		return self.compute_output(merged_input)

	def __call__(self, y, z, test=False, apply_f=False):
		self.test = test
		output = self.forward_one_step(y, z)
		if apply_f:
			return F.sigmoid(output)
		return output