# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2018/12/15 下午5:31
# @Project: unified_chinese_multi_tasking_framework
# @File: srl.py
# @Software: PyCharm

import tensorflow as tf
import transformer
import nn_utils
import train_utils
from lazy_adam_v2 import LazyAdamOptimizer


def pred_srl_op(concat_outputs_pos, concat_outputs_dep, tokens_to_keep,
                model_config, hparams, vocab,
                t_pre, t_srl, mode, pos_num_heads, dep_num_heads,
                pos_probabilities, pos_targets, rel_targets, head_targets, head_probabilities,
                rel_probabilities, embeddings, transition_params,
                pre_global_step, srl_global_step):
    with tf.variable_scope('predicate'):
        inputs_pre = concat_outputs_pos
        if mode == 'TRAIN' or mode == 'PREDICT':

            pos_probabilities_embedding = nn_utils.label_embedding_fn(pos_targets, embeddings['gold_pos'])
        else:
            pos_probabilities_embedding = nn_utils.label_embedding_fn(pos_probabilities, embeddings['gold_pos'])
        pre_num_heads = pos_num_heads + 1
        outputs_pre = transformer.transformer(inputs_pre, tokens_to_keep,
                                              model_config['layers']['head_dim'],
                                              pre_num_heads, hparams.attn_dropout,
                                              hparams.ff_dropout, hparams.prepost_dropout,
                                              model_config['layers']['ff_hidden_size'], [], 1,
                                              [pos_probabilities_embedding])

        outputs_pre = nn_utils.layer_norm(outputs_pre)
        pre_num_lables = vocab.vocab_names_sizes['predicate']
        loss_pre, pre_predictions, pre_scores, pre_probabilites \
            = nn_utils.softmax_classifier(hparams=hparams, inputs=outputs_pre,
                                          tokens_to_keep=tokens_to_keep, targets=t_pre,
                                          num_labels=pre_num_lables)
        loss_pre *= hparams.pre_penalty
        loss_pre += tf.reduce_sum([hparams.reg_lambda * tf.nn.l2_loss(x)
                                   for x in tf.trainable_variables()])

        loss_pre = tf.check_numerics(loss_pre, 'pred loss nan')
        with tf.control_dependencies([tf.no_op()]):
            lr = train_utils.learning_rate(hparams, pre_global_step)
            optimizer = LazyAdamOptimizer(learning_rate=lr, beta1=hparams.beta1,
                                          beta2=hparams.beta2, epsilon=hparams.epsilon,
                                          use_nesterov=hparams.use_nesterov)
            gradients, variables = zip(*optimizer.compute_gradients(loss_pre))
            gradients, _ = tf.clip_by_global_norm(gradients, hparams.gradient_clip_norm)
            optimize_op_pre = optimizer.apply_gradients(zip(gradients, variables),
                                                        global_step=pre_global_step)

    with tf.variable_scope('srl'):
        inputs_srl = concat_outputs_dep
        if mode == 'TRAIN' or mode == 'PREDICT':
            rel_probabilities_embedding = nn_utils.label_embedding_fn(rel_targets,
                                                                      embeddings['parse_label'])
            head_scores = [head_targets]
        else:
            head_scores = [head_probabilities]
            rel_probabilities_embedding = nn_utils.label_embedding_fn(rel_probabilities,
                                                                      embeddings['parse_label'])
        # rel_probabilities_embedding = tf.concat()
        srl_num_heads = dep_num_heads + 2
        outputs_srl = transformer.transformer(inputs_srl, tokens_to_keep,
                                              model_config['layers']['head_dim'],
                                              srl_num_heads, hparams.attn_dropout,
                                              hparams.ff_dropout, hparams.prepost_dropout,
                                              model_config['layers']['ff_hidden_size'], head_scores, 5,
                                              [pos_probabilities_embedding, rel_probabilities_embedding])
        outputs_srl = nn_utils.layer_norm(outputs_srl)
        srl_num_labels = vocab.vocab_names_sizes['srl']
        loss_srl, srl_predictions, srl_scores, srl_targets \
            = nn_utils.srl_bilinear_classifier(mode=mode, hparams=hparams, inputs=outputs_srl,
                                               targets=t_srl, num_labels=srl_num_labels,
                                               tokens_to_keep=tokens_to_keep,
                                               predicate_preds_train=t_pre,
                                               predicate_preds_eval=pre_predictions, predicate_targets=t_pre,
                                               transition_params=transition_params)
        loss_srl *= hparams.srl_penalty
        loss_srl += tf.reduce_sum(
            [hparams.reg_lambda * tf.nn.l2_loss(x) for x in tf.trainable_variables()])
        loss_srl = tf.check_numerics(loss_srl, 'srl grad nan')
        with tf.control_dependencies([tf.no_op()]):
            lr = train_utils.learning_rate(hparams, srl_global_step)

            optimizer = LazyAdamOptimizer(learning_rate=lr, beta1=hparams.beta1,
                                          beta2=hparams.beta2, epsilon=hparams.epsilon,
                                          use_nesterov=hparams.use_nesterov)
            gradients, variables = zip(*optimizer.compute_gradients(loss_srl))
            gradients, _ = tf.clip_by_global_norm(gradients, hparams.gradient_clip_norm)
            optimize_op_srl = optimizer.apply_gradients(zip(gradients, variables),
                                                        global_step=srl_global_step)

    return [optimize_op_pre, optimize_op_srl,
            loss_pre, loss_srl,
            pre_predictions, srl_predictions, srl_targets, lr]
