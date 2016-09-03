# -*- coding: utf-8 -*-
from args import args
from adgm import ADGM, Conf

conf = Conf()
conf.gpu_enabled = True if args.gpu_enabled == 1 else False

conf.learning_rate = 0.001
conf.gradient_momentum = 0.9
conf.gradient_clipping = 5.0
conf.wscale = 1

conf.ndim_z = 100
conf.ndim_a = 100
conf.n_mc_samples = args.n_mc_samples

conf.batchnorm_before_activation = True

conf.encoder_x_a_hidden_units = [500]
conf.encoder_x_a_activation_function = "elu"
conf.encoder_x_a_apply_dropout = False
conf.encoder_x_a_apply_batchnorm = True
conf.encoder_x_a_apply_batchnorm_to_input = True

conf.encoder_xy_z_hidden_units = [500]
conf.encoder_xy_z_activation_function = "elu"
conf.encoder_xy_z_apply_dropout = False
conf.encoder_xy_z_apply_batchnorm = True
conf.encoder_xy_z_apply_batchnorm_to_input = True

conf.encoder_ax_y_hidden_units = [500]
conf.encoder_ax_y_activation_function = "elu"
conf.encoder_ax_y_apply_dropout = False
conf.encoder_ax_y_apply_batchnorm = True
conf.encoder_ax_y_apply_batchnorm_to_input = True

conf.decoder_hidden_units = [500]
conf.decoder_activation_function = "elu"
conf.decoder_apply_dropout = False
conf.decoder_apply_batchnorm = True
conf.decoder_apply_batchnorm_to_input = True

adgm = ADGM(conf, name="adgm")
adgm.load(args.model_dir)