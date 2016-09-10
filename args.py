# -*- coding: utf-8 -*-
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_enabled", type=int, default=1)
parser.add_argument("--train_image_dir", type=str, default="../train_images")
parser.add_argument("--test_image_dir", type=str, default="../test_images")
parser.add_argument("--model_dir", type=str, default="model")
parser.add_argument("--plot_dir", type=str, default="plot")

# for semi-supervised learning
parser.add_argument("--n_labeled_data", type=int, default=100)

# seed for labeled / unlabeled split (this is used in semi-supervised learning)
parser.add_argument("--seed_ssl_split", type=int, default=None)

# seed
parser.add_argument("--seed", type=int, default=None)

parser.add_argument("--n_validation_data", type=int, default=10000)
parser.add_argument("--n_mc_samples", type=int, default=1)
args = parser.parse_args()