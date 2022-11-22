# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2018/12/15 下午5:31
# @Project: unified_chinese_multi_tasking_framework
# @File: cws_pos.py
# @Software: PyCharm

import tensorflow as tf
import transformer
import nn_utils
import train_utils
from lazy_adam_v2 import LazyAdamOptimizer


def pos_op(embeds, tokens_to_keep, model_config, hparams, vocab, t_pos, pos_global_step):
    with tf.variable_scope('pos'):
        pos_num_heads = model_config['layers']['num_heads']
        outputs_pos = transformer.transformer(embeds, tokens_to_keep, model_config['layers']['head_dim'],
                                              pos_num_heads, hparams.attn_dropout,
                                              hparams.ff_dropout, hparams.prepost_dropout,
                                              model_config['layers']['ff_hidden_size'], [], 3)

        # num_tokens = tf.reduce_sum(tokens_to_keep)

        concat_outputs_pos = outputs_pos
        outputs_pos = nn_utils.layer_norm(outputs_pos)
        pos_num_labels = vocab.vocab_names_sizes['gold_pos']
        loss_pos, pos_predictions, pos_scores, pos_probabilities \
            = nn_utils.softmax_classifier(hparams=hparams, inputs=outputs_pos, targets=t_pos,
                                          num_labels=pos_num_labels, tokens_to_keep=tokens_to_keep)

        loss_pos *= hparams.pos_penalty
        loss_pos += tf.reduce_sum([hparams.reg_lambda * tf.nn.l2_loss(x)
                                   for x in tf.trainable_variables()])
        loss_pos = tf.check_numerics(loss_pos, 'pos loss nan')
        with tf.control_dependencies([tf.no_op()]):
            lr = train_utils.learning_rate(hparams, pos_global_step)
            optimizer = LazyAdamOptimizer(learning_rate=lr, beta1=hparams.beta1,
                                          beta2=hparams.beta2, epsilon=hparams.epsilon,
                                          use_nesterov=hparams.use_nesterov)
            gradients, variables = zip(*optimizer.compute_gradients(loss_pos))
            gradients, _ = tf.clip_by_global_norm(gradients, hparams.gradient_clip_norm)
            optimize_op_pos = optimizer.apply_gradients(zip(gradients, variables),
                                                        global_step=pos_global_step)
        return [optimize_op_pos, concat_outputs_pos, loss_pos,
                pos_predictions, pos_scores, pos_probabilities, pos_num_heads, lr]

# todo cws&pos => transformer + crf
# add transition params
# nn_utils.joint_softmax_classifier()
