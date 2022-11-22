# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2019/1/27 下午4:45
# @Project: unified_chinese_multi_tasking_framework
# @File: train.py
# @Software: PyCharm

# S&T: CTB 4-7 \ SIGHAN \ CONLL
# Predicate: CONLL
# CHUNK: CTB 4, 5 (6-7 要修改ChunkLinkCTB, conll 用NLTK)
# Dependency: CTB 4-7, CONLL
# SRL: CONLL

import tensorflow as tf
import argparse
import os
from functools import partial
import train_utils
from vocab import Vocab
import numpy as np
import sys
import logging
from network import Network

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--train_files', required=True,
                        help='Comma-separated list of training data files')
arg_parser.add_argument('--dev_files', required=True,
                        help='Comma-separated list of development data files')
arg_parser.add_argument('--save_dir', required=True,
                        help='Directory to save models, outputs, etc.')
# todo load this more generically, so that we can have diff stats per task
arg_parser.add_argument('--transition_stats',
                        help='Transition statistics between labels')
arg_parser.add_argument('--hparams', type=str,
                        help='Comma separated list of "name=value" hyperparameter settings.')
arg_parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Whether to run in debug mode: a little faster and smaller')
arg_parser.add_argument('--data_config', required=True,
                        help='Path to data configuration json')
arg_parser.add_argument('--model_configs', required=True,
                        help='Comma-separated list of paths to model configuration json.')
arg_parser.add_argument('--task_configs', required=True,
                        help='Comma-separated list of paths to task configuration json.')
arg_parser.add_argument('--layer_configs', required=True,
                        help='Comma-separated list of paths to layer configuration json.')
arg_parser.add_argument('--attention_configs',
                        help='Comma-separated list of paths to attention configuration json.')
arg_parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Gpu_options.per_process_gpu_memory_fraction', default=0.9999)
arg_parser.set_defaults(debug=False)

args, leftovers = arg_parser.parse_known_args()
cargs = arg_parser.parse_args()

# Load all the various configurations
data_config = train_utils.load_json_configs(args.data_config)
model_config = train_utils.load_json_configs(args.model_configs)
task_config = train_utils.load_json_configs(args.task_configs, args)
layer_config = train_utils.load_json_configs(args.layer_configs)

print(layer_config)

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler(args.save_dir + 'tensorflow.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.log(tf.logging.INFO, "Using TensorFlow version %s" % tf.__version__)

hparams = train_utils.load_hparams(args, model_config)

#
# Set the random seed. This defaults to int(time.time()) if not otherwise set.
np.random.seed(hparams.random_seed)
tf.set_random_seed(hparams.random_seed)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

feature_idx_map = {}
label_idx_map = {}
for i, f in enumerate([d for d in data_config.keys() if
                       ('feature' in data_config[d] and data_config[d]['feature']) or
                       ('label' in data_config[d] and data_config[d]['label'])]):
    if 'feature' in data_config[f] and data_config[f]['feature']:
        feature_idx_map[f] = i
    if 'label' in data_config[f] and data_config[f]['label']:
        if 'type' in data_config[f] and data_config[f]['type'] == 'range':
            idx = data_config[f]['conll_idx']
            j = i + idx[1] if idx[1] != -1 else -1
            label_idx_map[f] = (i, j)
        else:
            label_idx_map[f] = (i, i + 1)
print('test')
print('feature_map:', feature_idx_map)
print('label_map', label_idx_map)
# train_desc = {'pos': 1000, 'dep': 2000, 'srl': 1000}
train_desc = {'pos': 1000, 'dep': 2500, 'srl': 2500}
# train_desc = {'pos': 1, 'dep': 2, 'srl': 2}

save_dir = args.save_dir
train_files = args.train_files
dev_files = args.dev_files

config_proto = tf.ConfigProto()
config_proto.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction

with tf.Graph().as_default()as graph:
    with tf.Session(config=config_proto, graph=graph) as sess:
        network = Network(hparams_read=hparams, model_config_read=model_config, data_config_read=data_config,
                          train_desc=train_desc, train_files=train_files, dev_files=dev_files, save_dir=save_dir,
                          mode='TRAIN')
        network.train(sess)

