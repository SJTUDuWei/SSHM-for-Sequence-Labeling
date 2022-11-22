# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2018/12/15 下午5:31
# @Project: unified_chinese_multi_tasking_framework
# @File: chunking.py
# @Software: PyCharm

from __future__ import division

import tensorflow as tf
from configures import Configures


class Chunking(Configures):
    def __init__(self):
        super(Chunking, self).__init__()