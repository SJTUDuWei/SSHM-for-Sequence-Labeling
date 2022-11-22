# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2018/12/15 下午5:30
# @Project: unified_chinese_multi_tasking_framework
# @File: dependency_parsing.py
# @Software: PyCharm

import tensorflow as tf
import transformer
import nn_utils
import train_utils
from lazy_adam_v2 import LazyAdamOptimizer


def dep_op(concat_outputs_pos, tokens_to_keep, model_config, hparams, vocab, t_head, t_dep, mode, pos_num_heads,
           pos_probabilities, pos_targets, embeddings, dep_global_step):
    with tf.variable_scope('dep'):
        inputs_dep = concat_outputs_pos
        if mode == 'TRAIN' or mode == 'PREDICT':
            pos_probabilities_embedding = nn_utils.label_embedding_fn(pos_targets,
                                                                  embeddings['gold_pos'])
        else:
            pos_probabilities_embedding = nn_utils.label_embedding_fn(pos_probabilities,
                                                                      embeddings['gold_pos'])
        dep_num_heads = pos_num_heads + 1
        outputs_dep = transformer.transformer(inputs_dep, tokens_to_keep,
                                              model_config['layers']['head_dim'],
                                              dep_num_heads, hparams.attn_dropout,
                                              hparams.ff_dropout, hparams.prepost_dropout,
                                              model_config['layers']['ff_hidden_size'], [],
                                              2, [pos_probabilities_embedding])
        outputs_dep = nn_utils.layer_norm(outputs_dep)
        concat_outputs_dep = outputs_dep
        head_loss, head_predictions, head_probabilites, head_scores, dep_rel_mlp, head_rel_mlp \
            = nn_utils.parse_bilinear(hparams=hparams, inputs=outputs_dep,
                                      targets=t_head, tokens_to_keep=tokens_to_keep)
        rel_num_labels = vocab.vocab_names_sizes['parse_label']
        rel_loss, rel_predictions, rel_probabilities, rel_scores \
            = nn_utils.conditional_bilinear(mode=mode, hparams=hparams, targets=t_dep,
                                            tokens_to_keep=tokens_to_keep, dep_rel_mlp=dep_rel_mlp,
                                            head_rel_mlp=head_rel_mlp, num_labels=rel_num_labels,
                                            parse_preds_train=t_head, parse_preds_eval=head_predictions)
        head_loss *= hparams.head_penalty
        rel_loss *= hparams.rel_penalty

        loss_dep = head_loss + rel_loss
        loss_dep += tf.reduce_sum([hparams.reg_lambda * tf.nn.l2_loss(x)
                                   for x in tf.trainable_variables()])

        loss_dep = tf.check_numerics(loss_dep, 'dep loss nan')
        with tf.control_dependencies([tf.no_op()]):
            lr = train_utils.learning_rate(hparams, dep_global_step)
            optimizer = LazyAdamOptimizer(learning_rate=lr, beta1=hparams.beta1,
                                          beta2=hparams.beta2, epsilon=hparams.epsilon,
                                          use_nesterov=hparams.use_nesterov)
            gradients, variables = zip(*optimizer.compute_gradients(loss_dep))
            gradients, _ = tf.clip_by_global_norm(gradients, hparams.gradient_clip_norm)
            optimize_op_dep = optimizer.apply_gradients(zip(gradients, variables), global_step=dep_global_step)

    return [optimize_op_dep, concat_outputs_dep, loss_dep,
            head_predictions, head_scores, head_probabilites,
            rel_predictions, rel_scores, rel_probabilities,
            dep_num_heads, lr]

