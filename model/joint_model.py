# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2018/12/15 下午5:30
# @Project: unified_chinese_multi_tasking_framework
# @File: joint_model.py
# @Software: PyCharm

# 参考LISA-V1  加入dependency parsing srl multi-head attention biaffie(bilinear classification)
# 修改数据格式

import numpy as np
import tensorflow as tf
from configures import Configures
from cws_pos import ChineseSegmentationPos
from chunking import Chunking
from dependency_parsing import DependencyParsing
from srl import SemanticRoleLabling
from ner import  NamedEntityRecognition
from lang_model import CharacterLanguageModel
from vocab import Vocab
from meta_lstm import *
from helper import *
# from tensorflow.estimator import ModeKeys
from layers import *
import losses

class JointMultiTaskModel(Configures):
    def __init__(self, num_tags, dim=200, lr=0.01,reg_lambda=0.001, mode='all'):
        super(JointMultiTaskModel, self).__init__()
        # self.train_hparams = hparams

        # self.lang_model = CharacterLanguageModel()
        self.lang_model = Vocab()
        self.segmentation_tagger = ChineseSegmentationPos()
        self.chunk = Chunking()
        self.named_entity = NamedEntityRecognition()
        self.dependency = DependencyParsing()
        self.semantic_role = SemanticRoleLabling()
        self.lr = lr
        self.dim = dim


        self.mode = mode
        self.reg_lambda = reg_lambda

        self.num_tag = num_tags
        self.transition_char = []
        for i in range(len(self.num_tag)):
            self.transition_char.append(tf.get_variable('transition_char' + str(i), [self.num_tag[i] + 1,
                                                                                     self.num_tag[i] + 1]))
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

    '''
    def get_seg_pos(self, sentence):
        return self.segmentation_tagger(sentence)

    def get_chunk(self, sentence, seg_pos, h_seg_pos):
        return self.chunk(sentence,  seg_pos, h_seg_pos)

    def get_dependency(self, sentence,  seg_pos, h_seg_pos, chunks, h_chunks):
        return self.dependency(sentence,  seg_pos, h_seg_pos, chunks, h_chunks)

    def get_ner(self, sentence, seg_pos, h_seg_pos, chunks, h_chunks, dep, h_dep):
        return self.named_entity(sentence,seg_pos, h_seg_pos, chunks, h_chunks, dep, h_dep)

    def get_srl(self, sentence, seg_pos, h_seg_pos, chunks, h_chunks, dep, h_dep, ners, h_ners):
        return self.semantic_role(sentence,  seg_pos, h_seg_pos, chunks, h_chunks, dep, h_dep, ners, h_ners)

    def run_all(self, x):
        sentences = self.embedded_batch(x)
        for s in sentences:
            y_segpos, h_segpos = self.get_seg_pos(s)
            y_chk, h_chk = self.get_chunk(s, y_segpos, h_segpos)
            y_dep, h_dep = self.get_dependency(s, y_segpos, h_segpos, y_chk, h_chk)
            y_ner, h_ner = self.get_ner(s, y_segpos, h_segpos, y_chk, h_chk, y_dep, h_dep)
            y_srl, h_srl = self.get_srl(s, y_segpos, h_segpos, y_chk, h_chk, y_dep, h_dep, y_ner, h_ner)
            yield y_segpos, y_chk, y_dep, y_ner, y_srl

    def forward(self, x):
        out = self.run_all(x)
        out = list(out)

        return out

    def segpos_loss(self, y, gold):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y, gold)\
               + (self.segmentation_tagger.w.norm() ** 2) * self.segpos_reg
        return loss
    
    '''

    def load_data(self):
        data = np.load('data/data01.npz')['data'].item()
        self.sent = data['sent'].values
        # self.seg = data['seg']
        # self.pos = data['pos']
        self.spp = data['spp']
        self.i2p = data['i2p']
        self.i2c = data['i2c']
        self.chun = data['chunk']
        self.dep = data['dep']

        self.w2i = data['w2i']

        self.vec = np.array(data['vec'] + [[0] * 300])
        self.max_length = max([len(i) for i in self.sent])
        print('***Data loaded***')

    def build_model(self):
        ''' Builds the whole computational graph '''

        def sentence_op(inputs, t_spp, t_chunk, t_dep, t_srl):

            with tf.variable_scope('segpospredicate'):
                embeddings = tf.constant(self.vec, dtype=tf.float32)
                embeds = tf.nn.embedding_lookup(embeddings, inputs)
                # fw_lstm = MyLSTM(self.dim, state_is_tuple=True)
                # bw_lstm = MyLSTM(self.dim, state_is_tuple=True)
                meta_fw_lstm, meta_bw_lstm = shared_lstm()
                lstm_fw_cell = MetaLSTM(self.dim, meta_fw_lstm)
                lstm_bw_cell = MetaLSTM(self.dim, meta_bw_lstm)
                outputs_segpos, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=embeds,
                                                             sequence_length=length(embeds), dtype=tf.float32)
                concat_outputs_spp = tf.concat(axis=2, values=outputs_segpos)
                tag_output_wrapper_spp = TimeDistributed(HiddenLayer(self.dim * 2, len(self.i2p), activation='linear',
                                                                 name='tag_hidden'), name='tag_output_wrapper_spp')
                y_spp = tag_output_wrapper_spp(concat_outputs_spp)
                t_spp_sparse = tf.one_hot(indices=t_spp, depth=len(self.i2p), axis=-1)
                loss = losses.loss_wrapper(y_spp, t_spp_sparse, losses.crf_loss, transitions=self.transition_char,
                                           nums_tags=self.num_tag, batch_size=self.batch_size)
                loss += tf.reduce_sum([self.reg_lambda * tf.nn.l2_loss(x)
                                       for x in tf.trainable_variables()])
                optimize_op_spp = tf.train.AdagradOptimizer(
                    self.lr).minimize(loss)

            with tf.variable_scope('chunk'):
                inputs_chunk = tf.concat(2, [embeds, concat_outputs_spp, y_spp])
                meta_fw_lstm, meta_bw_lstm = shared_lstm()
                lstm_fw_cell = MetaLSTM(self.dim, meta_fw_lstm)
                lstm_bw_cell = MetaLSTM(self.dim, meta_bw_lstm)
                outputs_chunk, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=inputs_chunk,
                                                             sequence_length=length(embeds), dtype=tf.float32)
                concat_outputs_chunk = tf.concat(2, outputs_chunk)
                y_chunk = activate(concat_outputs_chunk, [
                    self.dim * 2, len(self.i2c)], [len(self.i2c)])
                t_chunk_sparse = tf.one_hot(
                    indices=t_chunk, depth=len(self.i2c), axis=-1)
                loss_chunk = cost(y_chunk, t_chunk_sparse)
                loss_chunk += tf.reduce_sum([self.reg_lambda * tf.nn.l2_loss(x)
                                        for x in tf.trainable_variables()])
                optimize_op_chunk = tf.train.AdagradOptimizer(
                    self.lr).minimize(loss_chunk)

            # with tf.variable_scope('dep'):


            # with tf.variable_scope('srl'):


            return optimize_op_spp, optimize_op_chunk, loss, loss_chunk, y_spp, y_chunk

        def shared_lstm():
            with tf.variable_scope('shared_all_tasks') as scope:
                meta_fw_lstm = MyLSTM(self.dim, state_is_tuple=False)
                meta_bw_lstm = MyLSTM(self.dim, state_is_tuple=False)
                scope.reuse_variables()
            return meta_fw_lstm, meta_bw_lstm

        with tf.variable_scope('sentence') as scope:
            self.inp = tf.placeholder(
                shape=[None, self.max_length], dtype=tf.int32, name='input')
            self.t_p = tf.placeholder(
                shape=[None, self.max_length], dtype=tf.int32, name='t_pos')
            self.t_c = tf.placeholder(
                shape=[None, self.max_length], dtype=tf.int32, name='t_chunk')
            self.optimize_op_spp, self.optimize_op_chunk, self.loss, self.loss_chunk, self.y_spp, self.y_chunk = sentence_op(
                self.inp, self.t_p, self.t_c)
            scope.reuse_variables()
            self.inp1 = tf.placeholder(
                shape=[None, self.max_length], dtype=tf.int32, name='input1')
        print('***Model built***')

    def get_predictions(self, graph, task_desc):
        resp = dict()
        saver = tf.train.Saver()
        with tf.Session(graph=graph) as sess:
            saver = tf.train.import_meta_graph('saves/model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./saves'))

            if 'pos' in task_desc:
                inp = task_desc['pos'].lower().split()
                inputs = [[self.w2i[i] for i in inp] +
                          [self.vec.shape[0] - 1] * (self.max_length - len(inp))]
                preds = sess.run(self.y_spp,
                                 {self.inp: inputs})[0]
                preds = np.argmax(preds, axis=-1)[:len(inp)]
                preds = [self.i2p[i] for i in preds]
                resp['pos'] = preds

            if 'chunk' in task_desc:
                inp = task_desc['chunk'].lower().split()
                inputs = [[self.w2i[i] for i in inp] +
                          [self.vec.shape[0] - 1] * (self.max_length - len(inp))]
                preds = sess.run(self.y_chunk, {self.inp: inputs})[0]
                preds = np.argmax(preds, axis=-1)[:len(inp)]
                preds = [self.i2c[i] for i in preds]
                resp['chunk'] = preds

        return resp

    def train_model(self, graph, train_desc, resume=False):
        # 参考mep修改batch 和 predict 以及 metric输出
        saver = tf.train.Saver()
        batch_size = train_desc['batch_size']
        with tf.Session(graph=graph) as sess:
            if resume:
                saver = tf.train.import_meta_graph('saves/model.ckpt.meta')
                saver.restore(sess, tf.train.latest_checkpoint('./saves'))
                print('training resumed')
            else:
                sess.run(tf.global_variables_initializer())
            if 'pos' in train_desc:
                print('***Training CWS&POS&PREDICATE layer***')
                for i in range(train_desc['spp']):
                    a, b, c = get_batch_spp(self, batch_size)
                    _, l = sess.run([self.optimize_op_spp, self.loss],
                                    {self.inp: a, self.t_p: b})
                    if i % 50 == 0:
                        print(l)
                        saver.save(sess, 'saves/model.ckpt')
            if 'chunk' in train_desc:
                print('***Training chunk layer***')
                for i in range(train_desc['chunk']):
                    a, b, c = get_batch_spp(self, batch_size)
                    _, l1 = sess.run([self.optimize_op_chunk, self.loss_chunk], {
                        self.inp: a, self.t_p: b, self.t_c: c})
                    if i % 50 == 0:
                        print(l1)
                        saver.save(sess, 'saves/model.ckpt')

    def decode_graph(self):
        self.decode_holder = []
        self.score = []
        for nt in self.num_tags:
            ob = tf.placeholder(tf.float32, [None, self.max_length, nt])
            trans = tf.placeholder(tf.float32, [nt + 1, nt + 1])
            nums_steps = ob.get_shape().as_list()[1]
            length = tf.placeholder(tf.int32, [None])
            b_size = tf.placeholder(tf.int32, [])
            small = -1000
            class_pad = tf.stack(small * tf.ones([b_size, nums_steps, 1]))
            observations = tf.concat(axis=2, values=[ob, class_pad])
            b_vec = tf.tile(([small] * nt + [0]), [b_size])
            b_vec = tf.cast(b_vec, tf.float32)
            b_vec = tf.reshape(b_vec, [b_size, 1, -1])
            e_vec = tf.tile(([0] + [small] * nt), [b_size])
            e_vec = tf.cast(e_vec, tf.float32)
            e_vec = tf.reshape(e_vec, [b_size, 1, -1])
            observations = tf.concat(axis=1, values=[b_vec, observations, e_vec])
            transitions = tf.reshape(tf.tile(trans, [b_size, 1]), [b_size, nt + 1, nt + 1])
            observations = tf.reshape(observations, [-1, nums_steps + 2, nt + 1, 1])
            observations = tf.transpose(observations, [1, 0, 2, 3])
            previous = observations[0, :, :, :]
            max_scores = []
            max_scores_pre = []
            alphas = [previous]
            for t in range(1, nums_steps + 2):
                previous = tf.reshape(previous, [-1, nt + 1, 1])
                current = tf.reshape(observations[t, :, :, :], [-1, 1, nt + 1])
                alpha_t = previous + current + transitions
                max_scores.append(tf.reduce_max(alpha_t, axis=1))
                max_scores_pre.append(tf.argmax(alpha_t, axis=1))
                alpha_t = tf.reshape(Forward.log_sum_exp(alpha_t, axis=1), [-1, nt + 1, 1])
                alphas.append(alpha_t)
                previous = alpha_t
            max_scores = tf.stack(max_scores, axis=1)
            max_scores_pre = tf.stack(max_scores_pre, axis=1)
            self.decode_holder.append([ob, trans, length, b_size])
            self.score.append((max_scores, max_scores_pre))







