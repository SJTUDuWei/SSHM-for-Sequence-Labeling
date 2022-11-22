# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2018/12/17 下午11:51
# @Project: unified_chinese_multi_tasking_framework
# @File: constants.py
# @Software: PyCharm

import time

PAD_VALUE = -1
JOINT_LABEL_SEP = '/'

OOV_STRING = "<OOV>"

DEFAULT_BUCKET_BOUNDARIES = [20, 30, 50, 80]

VERY_LARGE = 1e9
VERY_SMALL = -1e9

# Optimizer hyperparameters
hparams = {
    'learning_rate': 0.04,
    'decay_rate': 1.5,
    'decay_steps': 5000,
    'warmup_steps': 8000,
    'beta1': 0.9,
    'beta2': 0.98,
    'epsilon': 1e-12,
    'use_nesterov': True,
    'batch_size': 256,
    'shuffle_buffer_multiplier': 20,
    'eval_throttle_secs': 1000,
    'eval_every_steps': 10,
    'num_train_epochs': 10000,
    'gradient_clip_norm': 5.0,
    'label_smoothing': 0.1,
    'moving_average_decay': 0.999,
    'average_norms': False,
    'input_dropout': 1.0,
    'bilinear_dropout': 1.0,
    'mlp_dropout': 1.0,
    'attn_dropout': 1.0,
    'ff_dropout': 1.0,
    'prepost_dropout': 1.0,
    'random_seed': int(time.time()),
    'validate_batch_size': 256,
    'pos_penalty': 1.0,
    'head_penalty': 1.0,
    'rel_penalty': 0.2,
    'pre_penalty': 1.0,
    'srl_penalty': 1.0,
    'reg_lambda': 0.00001,
    'predicate_mlp_size': 200,
    'role_mlp_size': 200,
    'predicate_pred_mlp_size': 200,
    'class_mlp_size': 100,
    'attn_mlp_size': 500,
    'print_every': 5,
    'save_every': 10,
    'validate_every': 50
}


def get_default(name):
    try:
        return hparams[name]
    except KeyError:
        print('Undefined default hparam value `%s' % name)
        exit(1)
