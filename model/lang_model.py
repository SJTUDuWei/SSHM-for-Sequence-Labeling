# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2018/12/15 下午11:46
# @Project: unified_chinese_multi_tasking_framework
# @File: lang_model.py
# @Software: PyCharm


from __future__ import division

import tensorflow as tf
from configures import Configures


class CharacterLanguageModel(Configures):
    def __init__(self):
        super(CharacterLanguageModel, self).__init__()

