# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2018/12/7 下午11:23
# @Project: unified_chinese_multi_tasking_framework
# @File: load_data.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import  os
import argparse
import numpy as np
import configparser


class Configures(object):
    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', type(self).__name__)
        if args and kwargs:
            raise TypeError('Configures must take either a config parser or keyword args!!!')
        if args:
            if len(args) > 1:
                raise TypeError('Configuers must at most one argument!!')
            self._config = args[0]
        else:
            self._config = self._configure(**kwargs)
        return

    def _configure(self, **kwargs):
        config = configparser.ConfigParser()
        config_f = [os.path.join('configure', 'defaults.cfg'),
                    os.path.join('configure', self.name.lower() + '.cfg'),
                    kwargs.pop('cfg_file', '')]
        config.read(config_f)
        for opt, val in kwargs.items():
            assigned = False
            for section in config.sections():
                if opt in config.options(section):
                    config.set(section, opt, str(val))
                    assigned = True
                    break
            if not assigned:
                raise ValueError('%s is not a valid option.' % opt)
        return config

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_file')
    arg_parser.add_argument('--data_dir')
    arg_parser.add_argument('--embedding_dir')

    @property
    def name(self):
        return self._name
    arg_parser.add_argument('--name')

    @property
    def embedding_file(self):
        return self._config.get('OS', 'embedding_file')
    arg_parser.add_argument('--embedding_file')

    @property
    def word_file(self):
        return self._config.get('OS', 'word_file')
    arg_parser.add_argument('--word_file')

    @property
    def pos_file(self):
        return self._config.get('OS', 'pos_file')
    arg_parser.add_argument('--pos_file')

    @property
    def dep_file(self):
        return self._config.get('OS', 'dep_file')
    arg_parser.add_argument('--dep_file')

    @property
    def train_file(self):
        return self._config.get('OS', 'train_file')
    arg_parser.add_argument('--train_file')

    @property
    def dev_file(self):
        return self._config.get('OS', 'dev_file')
    arg_parser.add_argument('--dev_file')

    @property
    def test_file(self):
        return self._config.get('OS', 'test_file')
    arg_parser.add_argument('--test_file')

    @property
    def save_dir(self):
        return self._config.get('OS', 'save_file')
    arg_parser.add_argument('--save_file')


def read_ctb_files( data_dir, task, version=5):
    data_dir = os.path.join(data_dir, task)
    files_dir = os.path.join(data_dir, 'ctb' + str(version))
    files_list = ['train', 'test', 'dev']
    train_data, test_data, dev_data = [], [], []
    task_file_type_tsv = ['pos', 'st', 'seg', 'ner']
    task_file_type_txt = ['dep', 'srl']
    task_files_dir = os.path.join(files_dir, task)
    if os.path.isdir(task_files_dir):
        if task in task_file_type_tsv:
            if os.path.isfile(os.path.join(task_files_dir, f + '.tsv')):
                train_data['']

    return


if __name__ == '__main__':
    print('Loading data...')

    print('Data imports to the model...')