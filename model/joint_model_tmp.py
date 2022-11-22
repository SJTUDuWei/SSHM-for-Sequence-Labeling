# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2018/12/15 下午5:30
# @Project: unified_chinese_multi_tasking_framework
# @File: joint_model.py
# @Software: PyCharm

# 参考LISA-V1  加入dependency parsing srl multi-head attention biaffie(bilinear classification)
# 修改数据格式
#  todo refer to Parse_V1 train and eval process code

import numpy as np
import tensorflow as tf
# from configures import Configures
# from cws_pos import ChineseSegmentationPos
# from chunking import Chunking
# from dependency_parsing import DependencyParsing
# from srl import SemanticRoleLabling
# from ner import NamedEntityRecognition
# from lang_model import CharacterLanguageModel
from vocab import Vocab
from meta_lstm import *
from helper import *
# from tensorflow.estimator import ModeKeys
from layers import *
import losses
import nn_utils
import transformer
from lazy_adam_v2 import LazyAdamOptimizer
import train_utils

PAD_VALUE = -1


class CJTMModel:

    def __init__(self, hparams, model_config, data_config,
                 feature_idx_map, label_idx_map, vocab,
                 mode='TRAIN'):
        # self.train_hparams = hparams

        # self.lang_model = CharacterLanguageModel()
        # self.segmentation_tagger = ChineseSegmentationPos()
        # self.chunk = Chunking()
        # self.named_entity = NamedEntityRecognition()
        # self.dependency = DependencyParsing()
        # self.semantic_role = SemanticRoleLabling()
        # self.lr = lr
        # self.dim = dim
        self.pos_num_heads = None
        self.dep_num_heads = None
        self.hparams = hparams

        self.mode = mode
        # self.hparams.reg_lambda = reg_lambda

        self.feature_idx_map = feature_idx_map
        self.label_idx_map = label_idx_map
        self.vocab = vocab

        self.model_config = model_config
        self.data_config = data_config

        # Create embeddings tables, loading pre-trained if specified
        embeddings = {}
        for embedding_name, embedding_map in self.model_config['embeddings'].items():
            embedding_dim = embedding_map['embedding_dim']
            if 'pretrained_embeddings' in embedding_map:
                input_pretrained_embeddings = embedding_map['pretrained_embeddings']
                include_oov = True
                embedding_table = self.get_embedding_table(embedding_name, embedding_dim, include_oov,
                                                           pretrained_fname=input_pretrained_embeddings)
            else:
                num_embeddings = self.vocab.vocab_names_sizes[embedding_name]
                include_oov = self.vocab.oovs[embedding_name]
                embedding_table = self.get_embedding_table(embedding_name, embedding_dim, include_oov,
                                                           num_embeddings=num_embeddings)
            embeddings[embedding_name] = embedding_table
            tf.logging.log(tf.logging.INFO, "Created embeddings for '%s'." % embedding_name)
        self.embeddings = embeddings

        # self.num_tag = num_tags
        # self.transition_char = []
        # for i in range(len(self.num_tag)):
        #     self.transition_char.append(tf.get_variable('transition_char' + str(i), [self.num_tag[i] + 1,
        #                                                                              self.num_tag[i] + 1]))
        self.all_metrics = ['Precision', 'Recall', 'F1-score', 'True-Negative-Rate', 'Boundary-F1-score']

    # def hparams(self, mode):
    #     if mode == ModeKeys.TRAIN:
    #         return self.train_hparams
    #     return self.test_hparams

    # def embedded_batch(self, batch_x):
    #     sent = batch_x
    #     # 每个batch的嵌入数据
    #     # embedded_batch = self.lang_model(batch_x)
    #     #
    #     # for batch in embedded_batch:
    #     #     sent = np.zeros((1, len(batch), self.embedding_size), dtype=np.float32)
    #     #     for i, word in enumerate(batch):
    #     #         sent[0, i] = word.data.numpy()
    #     yield sent

    @staticmethod
    def load_transitions(transition_statistics, num_classes, vocab_map):
        transition_statistics_np = np.zeros((num_classes, num_classes))
        with open(transition_statistics, 'r') as f:
            for line in f:
                tag1, tag2, prob = line.split("\t")
                transition_statistics_np[vocab_map[tag1], vocab_map[tag2]] = float(prob)
        return transition_statistics_np

    def get_embedding_table(self, name, embedding_dim, include_oov, pretrained_fname=None, num_embeddings=None):

        with tf.variable_scope("%s_embeddings" % name):
            initializer = tf.random_normal_initializer()
            if pretrained_fname:
                pretrained_embeddings = self.load_pretrained_embeddings(pretrained_fname)
                initializer = tf.constant_initializer(pretrained_embeddings)
                pretrained_num_embeddings, pretrained_embedding_dim = pretrained_embeddings.shape
                if pretrained_embedding_dim != embedding_dim:
                    tf.logging.log(tf.logging.ERROR, "Pre-trained %s embedding dim does not match"
                                                     " specified dim (%d vs %d)." % (name,
                                                                                     pretrained_embedding_dim,
                                                                                     embedding_dim))
                if num_embeddings and num_embeddings != pretrained_num_embeddings:
                    tf.logging.log(tf.logging.ERROR, "Number of pre-trained %s embeddings does not match"
                                                     " specified number of embeddings (%d vs %d)." % (name,
                                                                                                      pretrained_num_embeddings,
                                                                                                      num_embeddings))
                num_embeddings = pretrained_num_embeddings

            embedding_table = tf.get_variable(name="embeddings", shape=[num_embeddings, embedding_dim],
                                              initializer=initializer)

            if include_oov:
                oov_embedding = tf.get_variable(name="oov_embedding", shape=[1, embedding_dim],
                                                initializer=tf.random_normal_initializer())
                embedding_table = tf.concat([embedding_table, oov_embedding], axis=0,
                                            name="embeddings_table")

            return embedding_table

    @staticmethod
    def load_pretrained_embeddings(pretrained_fname):
        tf.logging.log(tf.logging.INFO, "Loading pre-trained embedding file: %s" % pretrained_fname)

        # TODO: np.loadtxt refuses to work for some reason
        # pretrained_embeddings = np.loadtxt(self.args.word_embedding_file, usecols=range(1, word_embedding_size+1))
        pretrained_embeddings = []
        with open(pretrained_fname, 'r', encoding='utf-8', errors='ignore') as f:
            line_i = 1
            for line in f:
                if line_i:
                    line_i -= 1
                    continue
                split_line = line.rstrip().split(' ')
                try:
                    embedding = list(map(float, split_line[1:]))
                    if len(embedding) != 100:
                        print(len(embedding))
                        print(split_line[0])
                        continue
                except ValueError:
                    continue
                pretrained_embeddings.append(embedding)
        pretrained_embeddings = np.array(pretrained_embeddings, dtype=np.float32)
        return pretrained_embeddings

    def build_model(self):
        ''' Builds the whole computational graph '''

        def pos_op(embeds, t_pos, t_head, t_dep, tokens_to_keep):
            with tf.variable_scope('pos'):
                self.pos_num_heads = self.model_config['layers']['num_heads']
                outputs_pos = transformer.transformer(embeds, self.tokens_to_keep, self.model_config['layers']['head_dim'],
                                                      self.pos_num_heads, self.hparams.attn_dropout,
                                                      self.hparams.ff_dropout, self.hparams.prepost_dropout,
                                                      self.model_config['layers']['ff_hidden_size'], 2)
                concat_outputs_pos = outputs_pos
                outputs_pos = nn_utils.layer_norm(outputs_pos)
                pos_num_labels = self.vocab.vocab_names_sizes['gold_pos']
                loss_pos, pos_predictions, pos_scores, pos_probabilities \
                    = nn_utils.softmax_classifier(hparams=self.hparams, inputs=outputs_pos, targets=t_pos,
                                                  num_labels=pos_num_labels, tokens_to_keep=self.tokens_to_keep)

                loss_pos *= self.hparams.pos_penalty
                loss_pos += tf.reduce_sum([self.hparams.reg_lambda * tf.nn.l2_loss(x)
                                           for x in tf.trainable_variables()])
                self.pos_global_step = tf.Variable(0, trainable=False, name='pos_global_step')
                self.lr = train_utils.learning_rate(self.hparams, self.pos_global_step)
                optimizer = LazyAdamOptimizer(learning_rate=self.lr, beta1=self.hparams.beta1,
                                              beta2=self.hparams.beta2, epsilon=self.hparams.epsilon,
                                              use_nesterov=self.hparams.use_nesterov)
                gradients, variables = zip(*optimizer.compute_gradients(loss_pos))
                gradients, _ = tf.clip_by_global_norm(gradients, self.hparams.gradient_clip_norm)
                optimize_op_pos = optimizer.apply_gradients(zip(gradients, variables),
                                                            global_step=self.pos_global_step)

            with tf.variable_scope('dep'):
                inputs_dep = concat_outputs_pos
                pos_probabilities_embedding = nn_utils.label_embedding_fn(pos_probabilities,
                                                                          self.embeddings['gold_pos'])
                self.dep_num_heads = self.pos_num_heads + 1
                outputs_dep = transformer.transformer(inputs_dep, self.tokens_to_keep,
                                                      self.model_config['layers']['head_dim'],
                                                      self.dep_num_heads, self.hparams.attn_dropout,
                                                      self.hparams.ff_dropout, self.hparams.prepost_dropout,
                                                      self.model_config['layers']['ff_hidden_size'],
                                                      4, [pos_probabilities_embedding])
                outputs_dep = nn_utils.layer_norm(outputs_dep)
                concat_outputs_dep = outputs_dep
                head_loss, head_predictions, head_probabilites, head_scores, dep_rel_mlp, head_rel_mlp \
                    = nn_utils.parse_bilinear(hparams=self.hparams, inputs=outputs_dep,
                                              targets=t_head, tokens_to_keep=self.tokens_to_keep)
                rel_num_labels = self.vocab.vocab_names_sizes['parse_label']
                rel_loss, rel_predictions, rel_probabilities, rel_scores \
                    = nn_utils.conditional_bilinear(mode=self.mode, hparams=self.hparams, targets=t_dep,
                                                    tokens_to_keep=self.tokens_to_keep, dep_rel_mlp=dep_rel_mlp,
                                                    head_rel_mlp=head_rel_mlp, num_labels=rel_num_labels,
                                                    parse_preds_train=t_head, parse_preds_eval=head_predictions)
                head_loss *= self.hparams.head_penalty
                rel_loss *= self.hparams.rel_penalty

                loss_dep = head_loss + rel_loss
                loss_dep += tf.reduce_sum([self.hparams.reg_lambda * tf.nn.l2_loss(x)
                                           for x in tf.trainable_variables()])
                self.dep_global_step = tf.Variable(0, trainable=False, name='dep_global_step')
                self.lr = train_utils.learning_rate(self.hparams, self.dep_global_step)
                optimizer = LazyAdamOptimizer(learning_rate=self.lr, beta1=self.hparams.beta1,
                                              beta2=self.hparams.beta2, epsilon=self.hparams.epsilon,
                                              use_nesterov=self.hparams.use_nesterov)
                gradients, variables = zip(*optimizer.compute_gradients(loss_dep))
                gradients, _ = tf.clip_by_global_norm(gradients, self.hparams.gradient_clip_norm)
                optimize_op_dep = optimizer.apply_gradients(zip(gradients, variables),
                                                            global_step=self.dep_global_step)

            return optimize_op_pos, optimize_op_dep, \
                   loss_pos, loss_dep, \
                   pos_predictions, head_predictions, rel_predictions, \
                   pos_probabilities, head_probabilites, rel_probabilities, \
                   concat_outputs_pos, concat_outputs_dep

        # def seg_pos_op(inputs, t_sp, t_chunk, t_ner, t_head, t_dep):
        #     with tf.variable_scope('segpos'):
        #         batch_shape = tf.shape(inputs)
        #         batch_size = batch_shape[0]
        #         batch_seq_len = batch_shape[1]
        #         embeddings = tf.constant(self.vec, dtype=tf.float32)
        #         embeds = tf.nn.embedding_lookup(embeddings, inputs)
        #         embeds = tf.nn.dropout(embeds, self.hparams.input_dropout)
        #         with tf.variable_scope('project_input'):
        #             embeds = nn_utils.MLP(embeds, self.model_config['layers']['head_dim'] * self.model_config['layers'][
        #                 'num_heads'], n_splits=1)
        #
        #         # for masking out padding tokens
        #         self.tokens_to_keep = tf.where(tf.equal(inputs, PAD_VALUE),
        #                                   tf.zeros([batch_size, batch_seq_len]),
        #                                   tf.ones([batch_size, batch_seq_len]))
        #
        #         # meta_fw_lstm, meta_bw_lstm = shared_lstm()
        #         # lstm_fw_cell = MetaLSTM(self.dim, meta_fw_lstm)
        #         # lstm_bw_cell = MetaLSTM(self.dim, meta_bw_lstm)
        #         # outputs_segpos, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,
        #         #                                                     inputs=embeds,
        #         #                                                     sequence_length=length(embeds), dtype=tf.float32)
        #
        #         inputs_sp = transformer.add_timing_signal_1d(embeds)
        #         outputs_sp = transformer.transformer(inputs_sp, self.tokens_to_keep, self.model_config['layers']['head_dim'],
        #                                              self.model_config['layers']['num_heads'],
        #                                              self.hparams.attn_dropout,
        #                                              self.hparams.ff_dropout, self.hparams.prepost_dropout,
        #                                              self.model_config['layers']['ff_hidden_size'], [], [])
        #         outputs_sp = nn_utils.layer_norm(outputs_sp)
        #         concat_outputs_sp = tf.concat(axis=2, values=outputs_sp)
        #         tag_output_wrapper_sp = TimeDistributed(HiddenLayer(self.dim * 2, len(self.i2p), activation='linear',
        #                                                             name='tag_hidden'), name='tag_output_wrapper_sp')
        #         y_sp = tag_output_wrapper_sp(concat_outputs_sp)
        #         t_sp_sparse = tf.one_hot(indices=t_sp, depth=len(self.i2p), axis=-1)
        #         loss = losses.loss_wrapper(y_sp, t_sp_sparse, losses.crf_loss, transitions=self.transition_char,
        #                                    nums_tags=len(self.i2p), batch_size=batch_size)
        #         loss += tf.reduce_sum([self.hparams.reg_lambda * tf.nn.l2_loss(x)
        #                                for x in tf.trainable_variables()])
        #         self.lr = train_utils.learning_rate(self.hparams, tf.train.get_global_step())
        #         optimizer = LazyAdamOptimizer(learning_rate=self.lr, beta1=self.hparams.beta1,
        #                                       beta2=self.hparams.beta2, epsilon=self.hparams.epsilon,
        #                                       use_nesterov=self.hparams.use_nesterov)
        #         gradients, variables = zip(*optimizer.compute_gradients(loss))
        #         gradients, _ = tf.clip_by_global_norm(gradients, self.hparams.gradient_clip_norm)
        #         optimize_op_sp = optimizer.apply_gradients(zip(gradients, variables),
        #                                                    global_step=tf.train.get_global_step())
        #
        #     with tf.variable_scope('chunk'):
        #         inputs_chunk = tf.concat(axis=2, values=[embeds, concat_outputs_sp, y_sp])
        #         # meta_fw_lstm, meta_bw_lstm = shared_lstm()
        #         # lstm_fw_cell = MetaLSTM(self.dim, meta_fw_lstm)
        #         # lstm_bw_cell = MetaLSTM(self.dim, meta_bw_lstm)
        #         # outputs_chunk, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,
        #         #                                                    inputs=inputs_chunk,
        #         #                                                    sequence_length=length(embeds), dtype=tf.float32)
        #         outputs_chunk = transformer.transformer(inputs_chunk, self.tokens_to_keep,
        #                                                 self.model_config['layers']['head_dim'],
        #                                                 self.model_config['layers']['num_heads'],
        #                                                 self.hparams.attn_dropout,
        #                                                 self.hparams.ff_dropout, self.hparams.prepost_dropout,
        #                                                 self.model_config['layers']['ff_hidden_size'], [], [])
        #
        #         outputs_chunk = nn_utils.layer_norm(outputs_chunk)
        #         concat_outputs_chunk = tf.concat(axis=2, values=outputs_chunk)
        #         y_chunk = activate(concat_outputs_chunk, [
        #             self.dim * 2, len(self.i2c)], [len(self.i2c)])
        #         t_chunk_sparse = tf.one_hot(
        #             indices=t_chunk, depth=len(self.i2c), axis=-1)
        #         loss_chunk = cost(y_chunk, t_chunk_sparse)
        #         loss_chunk += tf.reduce_sum([self.hparams.reg_lambda * tf.nn.l2_loss(x)
        #                                      for x in tf.trainable_variables()])
        #         self.lr = train_utils.learning_rate(self.hparams, tf.train.get_global_step())
        #         optimizer = LazyAdamOptimizer(learning_rate=self.lr, beta1=self.hparams.beta1,
        #                                       beta2=self.hparams.beta2, epsilon=self.hparams.epsilon,
        #                                       use_nesterov=self.hparams.use_nesterov)
        #         gradients, variables = zip(*optimizer.compute_gradients(loss_chunk))
        #         gradients, _ = tf.clip_by_global_norm(gradients, self.hparams.gradient_clip_norm)
        #         optimize_op_chunk = optimizer.apply_gradients(zip(gradients, variables),
        #                                                       global_step=tf.train.get_global_step())
        #
        #     with tf.variable_scope('ner'):
        #         inputs_ner = tf.concat(2, [embeds, concat_outputs_chunk, y_sp, y_chunk])
        #         # meta_fw_lstm, meta_bw_lstm = shared_lstm()
        #         # lstm_fw_cell = MetaLSTM(self.dim, meta_fw_lstm)
        #         # lstm_bw_cell = MetaLSTM(self.dim, meta_bw_lstm)
        #         # outputs_ner, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,
        #         #                                                    inputs=inputs_ner,
        #         #                                                    sequence_length=length(embeds), dtype=tf.float32)
        #         outputs_ner = transformer.transformer(inputs_ner, self.tokens_to_keep,
        #                                               self.model_config['layers']['head_dim'],
        #                                               self.model_config['layers']['num_heads'],
        #                                               self.hparams.attn_dropout,
        #                                               self.hparams.ff_dropout, self.hparams.prepost_dropout,
        #                                               self.model_config['layers']['ff_hidden_size'], [], [])
        #
        #         outputs_ner = nn_utils.layer_norm(outputs_ner)
        #         concat_outputs_ner = tf.concat(axis=2, values=outputs_ner)
        #         y_ner = activate(concat_outputs_ner, [
        #             self.dim * 2, len(self.i2c)], [len(self.i2c)])
        #         t_ner_sparse = tf.one_hot(
        #             indices=t_ner, depth=len(self.i2c), axis=-1)
        #         loss_ner = cost(y_ner, t_ner_sparse)
        #         loss_ner += tf.reduce_sum([self.hparams.reg_lambda * tf.nn.l2_loss(x)
        #                                    for x in tf.trainable_variables()])
        #         self.lr = train_utils.learning_rate(self.hparams, tf.train.get_global_step())
        #         optimizer = LazyAdamOptimizer(learning_rate=self.lr, beta1=self.hparams.beta1,
        #                                       beta2=self.hparams.beta2, epsilon=self.hparams.epsilon,
        #                                       use_nesterov=self.hparams.use_nesterov)
        #         gradients, variables = zip(*optimizer.compute_gradients(loss_ner))
        #         gradients, _ = tf.clip_by_global_norm(gradients, self.hparams.gradient_clip_norm)
        #         optimize_op_ner = optimizer.apply_gradients(zip(gradients, variables),
        #                                                     global_step=tf.train.get_global_step())
        #
        #     with tf.variable_scope('dep'):
        #         inputs_dep = tf.concat(axis=2, values=[embeds, concat_outputs_ner, y_sp, y_chunk, y_ner])
        #         outputs_dep = transformer.transformer(inputs_dep, self.tokens_to_keep,
        #                                               self.model_config['layers']['head_dim'],
        #                                               self.model_config['layers']['num_heads'],
        #                                               self.hparams.attn_dropout,
        #                                               self.hparams.ff_dropout, self.hparams.prepost_dropout,
        #                                               self.model_config['layers']['ff_hidden_size'], [], [])
        #         outputs_dep = nn_utils.layer_norm(outputs_dep)
        #         concat_outputs_dep = tf.concat(axis=2, values=outputs_dep)
        #         head_loss, head_predictions, head_probabilites, head_scores, dep_rel_mlp, head_rel_mlp \
        #             = nn_utils.parse_bilinear(hparams=self.hparams, inputs=outputs_dep,
        #                                       targets=t_head, tokens_to_keep=self.tokens_to_keep)
        #         rel_loss, rel_predictions, rel_probabilities, rel_scores \
        #             = nn_utils.conditional_bilinear(mode=self.mode, hparams=self.hparams, targets=t_dep,
        #                                             tokens_to_keep=self.tokens_to_keep, dep_rel_mlp=dep_rel_mlp,
        #                                             head_rel_mlp=head_rel_mlp, num_labels=len(self.i2r),
        #                                             parse_preds_train=t_head, parse_preds_eval=head_predictions)
        #         head_loss *= self.hparams.head_penalty
        #         rel_loss *= self.hparams.rel_penalty
        #
        #         loss_dep = head_loss + rel_loss
        #         loss_dep += tf.reduce_sum([self.hparams.reg_lambda * tf.nn.l2_loss(x)
        #                                    for x in tf.trainable_variables()])
        #         self.lr = train_utils.learning_rate(self.hparams, tf.train.get_global_step())
        #         optimizer = LazyAdamOptimizer(learning_rate=self.lr, beta1=self.hparams.beta1,
        #                                       beta2=self.hparams.beta2, epsilon=self.hparams.epsilon,
        #                                       use_nesterov=self.hparams.use_nesterov)
        #         gradients, variables = zip(*optimizer.compute_gradients(loss_dep))
        #         gradients, _ = tf.clip_by_global_norm(gradients, self.hparams.gradient_clip_norm)
        #         optimize_op_dep = optimizer.apply_gradients(zip(gradients, variables),
        #                                                     global_step=tf.train.get_global_step())
        #
        #     return [optimize_op_sp, optimize_op_chunk, optimize_op_ner, optimize_op_dep, \
        #             loss, loss_chunk, loss_ner, loss_dep, \
        #             y_sp, y_chunk, y_ner, head_probabilites, rel_probabilities,
        #             concat_outputs_dep]

        def pre_srl_op(embeds, tokens_to_keep, concat_outputs_sp, y_sp,
                       head_probabilites, rel_probabilities, t_pre, t_srl,
                       concat_outputs_dep, transition_params):
            with tf.variable_scope('predicate'):
                # inputs_pre = tf.concat(axis=2, values=[embeds, concat_outputs_sp, y_sp])
                # with tf.variable_scope('project_input'):
                #     inputs_pre = nn_utils.MLP(inputs, self.model_config['layers']['head_dim'] * self.model_config['layers']['num_heads'], n_splits=1)
                #     inputs_pre = tf.concat(2, [inputs_pre, concat_outputs_sp, y_sp])
                # inputs_pre = transformer.add_timing_signal_1d(inputs_pre)
                inputs_pre = concat_outputs_sp
                y_sp_embedding = nn_utils.label_embedding_fn(y_sp, self.embeddings['gold_pos'])
                self.pre_num_heads = self.pos_num_heads + 1
                outputs_pre = transformer.transformer(inputs_pre, self.tokens_to_keep,
                                                      self.model_config['layers']['head_dim'],
                                                      self.pre_num_heads, self.hparams.attn_dropout,
                                                      self.hparams.ff_dropout, self.hparams.prepost_dropout,
                                                      self.model_config['layers']['ff_hidden_size'], 2,
                                                      [y_sp_embedding])

                outputs_pre = nn_utils.layer_norm(outputs_pre)
                pre_num_lables = self.vocab.vocab_names_sizes['predicate']
                loss_pre, pre_predictions, pre_scores, pre_probabilites \
                    = nn_utils.softmax_classifier(hparams=self.hparams, inputs=outputs_pre,
                                                  tokens_to_keep=self.tokens_to_keep, targets=t_pre,
                                                  num_labels=pre_num_lables)
                loss_pre *= self.hparams.pre_penalty
                loss_pre += tf.reduce_sum([self.hparams.reg_lambda * tf.nn.l2_loss(x)
                                           for x in tf.trainable_variables()])
                self.pre_global_step = tf.Variable(0, trainable=False, name='pre_global_step')
                self.lr = train_utils.learning_rate(self.hparams, self.pre_global_step)
                optimizer = LazyAdamOptimizer(learning_rate=self.lr, beta1=self.hparams.beta1,
                                              beta2=self.hparams.beta2, epsilon=self.hparams.epsilon,
                                              use_nesterov=self.hparams.use_nesterov)
                gradients, variables = zip(*optimizer.compute_gradients(loss_pre))
                gradients, _ = tf.clip_by_global_norm(gradients, self.hparams.gradient_clip_norm)
                optimize_op_pre = optimizer.apply_gradients(zip(gradients, variables),
                                                            global_step=self.pre_global_step)

            with tf.variable_scope('srl'):
                # inputs_srl = tf.concat(axis=2, values=[embeds, concat_outputs_dep, y_sp,
                #                            pre_probabilites, head_probabilites, rel_probabilities])
                inputs_srl = concat_outputs_dep
                rel_probabilities_embedding = nn_utils.label_embedding_fn(rel_probabilities,
                                                                          self.embeddings['parse_label'])
                self.srl_num_heads = self.dep_num_heads + 2
                outputs_srl = transformer.transformer(inputs_srl, self.tokens_to_keep,
                                                      self.model_config['layers']['head_dim'],
                                                      self.srl_num_heads, self.hparams.attn_dropout,
                                                      self.hparams.ff_dropout, self.hparams.prepost_dropout,
                                                      self.model_config['layers']['ff_hidden_size'], 4,
                                                      [y_sp_embedding, rel_probabilities_embedding])
                outputs_srl = nn_utils.layer_norm(outputs_srl)
                srl_num_labels = self.vocab.vocab_names_sizes['srl']
                loss_srl, srl_predictions, srl_scores, srl_targets \
                    = nn_utils.srl_bilinear_classifier(mode=self.mode, hparams=self.hparams, inputs=outputs_srl,
                                                       targets=t_srl, num_labels=srl_num_labels,
                                                       tokens_to_keep=self.tokens_to_keep,
                                                       predicate_preds_train=pre_predictions,
                                                       predicate_preds_eval=t_pre, predicate_targets=t_pre,
                                                       transition_params=transition_params)
                loss_srl *= self.hparams.srl_penalty
                loss_srl += tf.reduce_sum(
                    [self.hparams.reg_lambda * tf.nn.l2_loss(x) for x in tf.trainable_variables()])
                self.srl_global_step = tf.Variable(0, trainable=False, name='srl_global_step')
                self.lr = train_utils.learning_rate(self.hparams, self.srl_global_step)
                optimizer = LazyAdamOptimizer(learning_rate=self.lr, beta1=self.hparams.beta1,
                                              beta2=self.hparams.beta2, epsilon=self.hparams.epsilon,
                                              use_nesterov=self.hparams.use_nesterov)
                gradients, variables = zip(*optimizer.compute_gradients(loss_srl))
                gradients, _ = tf.clip_by_global_norm(gradients, self.hparams.gradient_clip_norm)
                optimize_op_srl = optimizer.apply_gradients(zip(gradients, variables),
                                                            global_step=self.srl_global_step)

                return optimize_op_pre, optimize_op_srl, \
                       loss_pre, loss_srl, \
                       pre_predictions, srl_predictions, srl_targets

        # def shared_lstm():
        #     with tf.variable_scope('shared_all_tasks') as scope:
        #         meta_fw_lstm = MyLSTM(self.dim, state_is_tuple=False)
        #         meta_bw_lstm = MyLSTM(self.dim, state_is_tuple=False)
        #         scope.reuse_variables()
        #     return meta_fw_lstm, meta_bw_lstm

        with tf.variable_scope('CJTM', reuse=tf.AUTO_REUSE):
            self.features = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name='features')
            batch_shape = tf.shape(self.features)
            batch_size = batch_shape[0]
            batch_seq_len = batch_shape[1]

            self.feats = {f: self.features[:, :, idx] for f, idx in self.feature_idx_map.items()}

            inputs = self.feats['word_type']

            self.tokens_to_keep = tf.where(tf.equal(inputs, PAD_VALUE), tf.zeros([batch_size, batch_seq_len]),
                                      tf.ones([batch_size, batch_seq_len]))

            self.feats = {f: tf.multiply(tf.cast(self.tokens_to_keep, tf.int32), v) for f, v in self.feats.items()}

            # inputs = tf.multiply(tf.cast(self.tokens_to_keep, tf.int32), inputs)

            # Extract named labels from monolithic "self.features" input, and mask them
            labels = {}
            for l, idx in self.label_idx_map.items():
                these_labels = self.features[:, :, idx[0]:idx[1]] if idx[1] != -1 else self.features[:, :, idx[0]:]
                these_labels_masked = tf.multiply(these_labels, tf.cast(tf.expand_dims(self.tokens_to_keep, -1), tf.int32))
                # check if we need to mask another dimension
                if idx[1] == -1:
                    last_dim = tf.shape(these_labels)[2]
                    this_mask = tf.where(tf.equal(these_labels_masked, PAD_VALUE),
                                         tf.zeros([batch_size, batch_seq_len, last_dim], dtype=tf.int32),
                                         tf.ones([batch_size, batch_seq_len, last_dim], dtype=tf.int32))
                    these_labels_masked = tf.multiply(these_labels_masked, this_mask)
                else:
                    these_labels_masked = tf.squeeze(these_labels_masked, -1)
                labels[l] = these_labels_masked

            self.labels = labels

            # Set up model inputs
            inputs_list = []
            for input_name in self.model_config['inputs']:
                input_values = self.feats[input_name]
                input_embedding_lookup = tf.nn.embedding_lookup(self.embeddings[input_name], input_values)
                inputs_list.append(input_embedding_lookup)
                tf.logging.log(tf.logging.INFO, "Added %s to inputs list." % input_name)
            current_input = tf.concat(inputs_list, axis=2)
            current_input = tf.nn.dropout(current_input, self.hparams.input_dropout)

            with tf.variable_scope('project_input'):
                current_input = nn_utils.MLP(current_input, self.model_config['layers']['head_dim'] *
                                             self.model_config['layers']['num_heads'])

            current_input = transformer.add_timing_signal_1d(current_input)

            self.optimize_op_pos, self.optimize_op_dep, self.loss_pos, self.loss_dep, self.pos_predictions, \
            self.head_predictions, self.rel_predictions, self.pos_probabilities, self.head_probabilites, \
            self.rel_probabilities, concat_outputs_pos, concat_outputs_dep = pos_op(embeds=current_input,
                                                                                    tokens_to_keep=self.tokens_to_keep,
                                                                                    t_pos=labels['gold_pos'],
                                                                                    t_head=labels['parse_head'],
                                                                                    t_dep=labels['parse_label'])
            # todo transitions param created by compute_transition_probs.py
            # generate transition_prob.tsv under data_dir
            srl_vocab_size = self.vocab.vocab_names_sizes['srl']

            transition_prob_np = self.load_transitions(
                transition_statistics=self.model_config['transition_stats_file'],
                num_classes=srl_vocab_size, vocab_map=self.vocab.vocab_maps['srl'])

            # crf
            transition_params = tf.get_variable("transition", shape=[srl_vocab_size, srl_vocab_size],
                                                initializer=tf.constant_initializer(transition_prob_np),
                                                trainable=True)

            self.optimize_op_pre, self.optimize_op_srl, self.loss_pre, self.loss_srl, self.pre_predictions,\
            self.srl_predictions, self.srl_targets = pre_srl_op(embeds=current_input, concat_outputs_sp=concat_outputs_pos,
                             concat_outputs_dep=concat_outputs_dep, tokens_to_keep=self.tokens_to_keep,
                             y_sp=self.pos_probabilities, head_probabilites=self.head_probabilites,
                             rel_probabilities=self.rel_probabilities, t_pre=labels['predicate'], t_srl=labels['srl'],
                             transition_params=transition_params)

        # return {'opt_pos': self.optimize_op_pos, 'opt_dep': self.optimize_op_dep,
        #         'opt_pre': self.optimize_op_pre, 'opt_srl': self.optimize_op_srl,
        #         'loss_pos': self.loss_pos, 'loss_dep': self.loss_dep,
        #         'loss_pre': self.loss_pre, 'loss_srl': self.loss_srl}
        print('***Model built***')

    # def get_predictions(self, graph, task_desc):
    #     resp = dict()
    #     saver = tf.train.Saver()
    #     with tf.Session(graph=graph) as sess:
    #         saver = tf.train.import_meta_graph('saves/model.ckpt.meta')
    #         saver.restore(sess, tf.train.latest_checkpoint('./saves'))
    #
    #         if 'pos' in task_desc:
    #             inp = task_desc['pos'].lower().split()
    #             inputs = [[self.w2i[i] for i in inp] +
    #                       [self.vec.shape[0] - 1] * (self.max_length - len(inp))]
    #             preds = sess.run(self.y_sp,
    #                              {self.inp: inputs})[0]
    #             preds = np.argmax(preds, axis=-1)[:len(inp)]
    #             preds = [self.i2p[i] for i in preds]
    #             resp['pos'] = preds
    #
    #         if 'chunk' in task_desc:
    #             inp = task_desc['chunk'].lower().split()
    #             inputs = [[self.w2i[i] for i in inp] +
    #                       [self.vec.shape[0] - 1] * (self.max_length - len(inp))]
    #             preds = sess.run(self.y_chunk, {self.inp: inputs})[0]
    #             preds = np.argmax(preds, axis=-1)[:len(inp)]
    #             preds = [self.i2c[i] for i in preds]
    #             resp['chunk'] = preds
    #
    #     return resp

    # def train_model(self, graph, train_desc, resume=False):
    #     # 参考mep修改batch 和 predict 以及 metric输出
    #     saver = tf.train.Saver()
    #     batch_size = train_desc['batch_size']
    #     with tf.Session(graph=graph) as sess:
    #         if resume:
    #             saver = tf.train.import_meta_graph('saves/model.ckpt.meta')
    #             saver.restore(sess, tf.train.latest_checkpoint('./saves'))
    #             print('training resumed')
    #         else:
    #             sess.run(tf.global_variables_initializer())
    #         if 'pos' in train_desc:
    #             print('***Training CWS&POS&PREDICATE layer***')
    #             for i in range(train_desc['spp']):
    #                 a, b, c = get_batch_sp(self, batch_size)
    #                 _, l = sess.run([self.optimize_op_sp, self.loss],
    #                                 {self.inp: a, self.t_p: b})
    #                 if i % 50 == 0:
    #                     print(l)
    #                     saver.save(sess, 'saves/model.ckpt')
    #         if 'chunk' in train_desc:
    #             print('***Training chunk layer***')
    #             for i in range(train_desc['chunk']):
    #                 a, b, c = get_batch_sp(self, batch_size)
    #                 _, l1 = sess.run([self.optimize_op_chunk, self.loss_chunk], {
    #                     self.inp: a, self.t_p: b, self.t_c: c})
    #                 if i % 50 == 0:
    #                     print(l1)
    #                     saver.save(sess, 'saves/model.ckpt')

    # def decode_graph(self):
    #     self.decode_holder = []
    #     self.score = []
    #     for nt in self.num_tags:
    #         ob = tf.placeholder(tf.float32, [None, self.max_length, nt])
    #         trans = tf.placeholder(tf.float32, [nt + 1, nt + 1])
    #         nums_steps = ob.get_shape().as_list()[1]
    #         length = tf.placeholder(tf.int32, [None])
    #         b_size = tf.placeholder(tf.int32, [])
    #         small = -1000
    #         class_pad = tf.stack(small * tf.ones([b_size, nums_steps, 1]))
    #         observations = tf.concat(axis=2, values=[ob, class_pad])
    #         b_vec = tf.tile(([small] * nt + [0]), [b_size])
    #         b_vec = tf.cast(b_vec, tf.float32)
    #         b_vec = tf.reshape(b_vec, [b_size, 1, -1])
    #         e_vec = tf.tile(([0] + [small] * nt), [b_size])
    #         e_vec = tf.cast(e_vec, tf.float32)
    #         e_vec = tf.reshape(e_vec, [b_size, 1, -1])
    #         observations = tf.concat(axis=1, values=[b_vec, observations, e_vec])
    #         transitions = tf.reshape(tf.tile(trans, [b_size, 1]), [b_size, nt + 1, nt + 1])
    #         observations = tf.reshape(observations, [-1, nums_steps + 2, nt + 1, 1])
    #         observations = tf.transpose(observations, [1, 0, 2, 3])
    #         previous = observations[0, :, :, :]
    #         max_scores = []
    #         max_scores_pre = []
    #         alphas = [previous]
    #         for t in range(1, nums_steps + 2):
    #             previous = tf.reshape(previous, [-1, nt + 1, 1])
    #             current = tf.reshape(observations[t, :, :, :], [-1, 1, nt + 1])
    #             alpha_t = previous + current + transitions
    #             max_scores.append(tf.reduce_max(alpha_t, axis=1))
    #             max_scores_pre.append(tf.argmax(alpha_t, axis=1))
    #             alpha_t = tf.reshape(Forward.log_sum_exp(alpha_t, axis=1), [-1, nt + 1, 1])
    #             alphas.append(alpha_t)
    #             previous = alpha_t
    #         max_scores = tf.stack(max_scores, axis=1)
    #         max_scores_pre = tf.stack(max_scores_pre, axis=1)
    #         self.decode_holder.append([ob, trans, length, b_size])
    #         self.score.append((max_scores, max_scores_pre))
