# # -*- coding: utf-8 -*-
# # @Author: Jie Zhou
# # @Time: 2018/12/10 上午12:02
# # @Project: unified_chinese_multi_tasking_framework
# # @File: parser.py
# # @Software: PyCharm
#
# from __future__ import  absolute_import
# from __future__ import division
#
# import numpy as np
# import tensorflow as tf
# from vocab import Vocab
# from models import NN
#
#
# class BaseParser(NN):
#     def __call__(self, dataset, moving_params=None):
#         raise  NotImplementedError
#
#     def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
#         raise NotImplementedError
#
#     def sanity_check(self, inputs, targets, predictions, vocabs, fileobject, feed_dict={}):
#         for tokens, golds, parse_preds, rel_preds in zip(inputs, targets, predictions[0], predictions[1]):
#             for l, (token, gold, parse, rel) in enumerate(zip(tokens, golds, parse_preds, rel_preds)):
#                 if token[0] > 0:
#                     word = vocabs[0][token[0]]
#                     glove = vocabs[0].get_embed(token[1])
#                     tag = vocabs[1][token[2]]
#                     gold_tag = vocabs[1][gold[0]]
#                     pred_parse = parse
#                     pred_rel = vocabs[2][rel]
#                     gold_parse = gold[1]
#                     gold_rel = vocabs[2][gold[2]]
#                     fileobject.write('%d\t%s\t%s\t%s\t%s\t_\t%d\t%s\t%d\t%s\n' %
#                                      (l, word, glove, tag, gold_tag, pred_parse, pred_rel, gold_parse, gold_rel))
#             fileobject.write('\n')
#         return
#
#     def validate(self, mb_inputs, mb_targets, mb_probs):
#
#         sents = []
#         mb_parse_probs, mb_rel_probs = mb_probs
#         for inputs, targets, parse_probs, rel_probs in zip(mb_inputs, mb_targets, mb_parse_probs, mb_rel_probs):
#             tokens_to_keep = np.greater(inputs[:, 0], Vocab.ROOT)
#             length = np.sum(tokens_to_keep)
#             parse_preds, rel_preds = self.prob_argmax(parse_probs, rel_probs, tokens_to_keep)
#
#             sent = -np.ones((length, 9), dtype=int)
#             tokens = np.arange(1, int(length + 1))
#             sent[:, 0] = tokens
#             sent[:, 1:4] = inputs[tokens]
#             sent[:, 4] = targets[tokens, 0]
#             sent[:, 5] = parse_preds[tokens]
#             sent[:, 6] = rel_preds[tokens]
#             sent[:, 7:] = targets[tokens, 1:]
#             sents.append(sent)
#         return sents
#
#     @staticmethod
#     def evaluate(filename, punct=NN.PUNCT):
#         correct = {'UAS': [], 'LAS': []}
#         with open(filename) as f:
#             for line in f:
#                 line = line.strip().split('\t')
#                 if len(line) == 10 and line[4] not in punct:
#                     correct['UAS'].append(0)
#                     correct['LAS'].append(0)
#                     if line[6] == line[8]:
#                         correct['UAS'][-1] = 1
#                         if line[7] == line[9]:
#                             correct['LAS'][-1] = 1
#         correct = {k: np.array(v) for k, v in correct.items()}
#         return 'UAS: %.2f    LAS: %.2f\n' % (float(np.mean(correct['UAS'])) * 100, float(np.mean(correct['LAS'])) * 100), correct
#
#     @property
#     def input_idxs(self):
#         return 0, 1, 2
#
#     @property
#     def target_idxs(self):
#         return 3, 4, 5
#
#
# class Parser(BaseParser):
#     def __call__(self, dataset, moving_params=None):
#         vocabs = dataset.vocabs
#         inputs = dataset.inputs
#         targets = dataset.targets
#
#         reuse = (moving_params is not None)
#         self.tokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs[:, :, 0], vocabs[0].ROOT)), 2)
#         self.sequence_lengths = tf.reshape(tf.reduce_sum(self.tokens_to_keep3D, [1, 2]), [-1, 1])
#         self.n_tokens = tf.reduce_sum(self.sequence_lengths)
#         self.moving_params = moving_params
#
#         word_inputs, pret_inputs = vocabs[0].embedding_lookup(inputs[:, :, 0], inputs[:, :, 1],
#                                                               moving_params=self.moving_params)
#         tag_inputs = vocabs[1].embedding_lookup(inputs[:, :, 2], moving_params=self.moving_params)
#         if self.add_to_pretrained:
#             word_inputs += pret_inputs
#         if self.word_l2_reg > 0:
#             unk_mask = tf.expand_dims(tf.to_float(tf.greater(inputs[:, :, 1], vocabs[0].UNK)), 2)
#             word_loss = self.word_l2_reg * tf.nn.l2_loss((word_inputs - pret_inputs) * unk_mask)
#         embed_inputs = self.embed_concat(word_inputs, tag_inputs)
#
#         top_recur = embed_inputs
#         for i in range(self.n_recur):
#             with tf.variable_scope('RNN%d' % i, reuse=reuse):
#                 top_recur, _ = self.RNN(top_recur)
#
#         with tf.variable_scope('MLP', reuse=reuse):
#             dep_mlp, head_mlp = self.MLP(top_recur, self.class_mlp_size + self.attn_mlp_size, n_splits=2)
#             dep_arc_mlp, dep_rel_mlp = dep_mlp[:, :, :self.attn_mlp_size], dep_mlp[:, :, self.attn_mlp_size:]
#             head_arc_mlp, head_rel_mlp = head_mlp[:, :, :self.attn_mlp_size], head_mlp[:, :, self.attn_mlp_size:]
#
#         with tf.variable_scope('Arcs', reuse=reuse):
#             arc_logits = self.bilinear_classifier(dep_arc_mlp, head_arc_mlp)
#             arc_output = self.output(arc_logits, targets[:, :, 1])
#             if moving_params is None:
#                 predictions = targets[:, :, 1]
#             else:
#                 predictions = arc_output['predictions']
#         with tf.variable_scope('Rels', reuse=reuse):
#             rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp,
#                                                                                len(vocabs[2]), predictions)
#             rel_output = self.output(rel_logits, targets[:, :, 2])
#             rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond)
#
#         output = {}
#         output['probabilities'] = tf.tuple([arc_output['probabilities'],
#                                             rel_output['probabilities']])
#         output['predictions'] = tf.pack([arc_output['predictions'],
#                                          rel_output['predictions']])
#         output['correct'] = arc_output['correct'] * rel_output['correct']
#         output['tokens'] = arc_output['tokens']
#         output['n_correct'] = tf.reduce_sum(output['correct'])
#         output['n_tokens'] = self.n_tokens
#         output['accuracy'] = output['n_correct'] / output['n_tokens']
#         output['loss'] = arc_output['loss'] + rel_output['loss']
#         if self.word_l2_reg > 0:
#             output['loss'] += word_loss
#
#         output['embed'] = embed_inputs
#         output['recur'] = top_recur
#         output['dep_arc'] = dep_arc_mlp
#         output['head_dep'] = head_arc_mlp
#         output['dep_rel'] = dep_rel_mlp
#         output['head_rel'] = head_rel_mlp
#         output['arc_logits'] = arc_logits
#         output['rel_logits'] = rel_logits
#         return output
#
#         # =============================================================
#
#     def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
#         """"""
#
#         parse_preds = self.parse_argmax(parse_probs, tokens_to_keep)
#         rel_probs = rel_probs[np.arange(len(parse_preds)), parse_preds]
#         rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
#         return parse_preds, rel_preds
#
