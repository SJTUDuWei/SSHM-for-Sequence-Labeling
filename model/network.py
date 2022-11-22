# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2019/1/28 下午3:10
# @Project: unified_chinese_multi_tasking_framework
# @File: network.py
# @Software: PyCharm

from joint_model_tmp2 import CJTMModel
from vocab import Vocab
import train_utils
import numpy as np
import tensorflow as tf
import os
import time
import sys
import evaluates
import pickle
import math


class Network:
    def __init__(self, hparams_read, model_config_read, data_config_read, train_desc,
                 save_dir, dev_files, train_files=None, test_files=None, mode='TRAIN'):
        self.hparams = hparams_read
        self.model_config = model_config_read
        self.data_config = data_config_read
        self.mode = mode

        self.save_dir = save_dir
        if train_files:
            self.train_filenames = train_files.split(',')
        self.dev_filenames = dev_files.split(',')
        if test_files:
            self.test_filenames = test_files.split(',')

        if train_files:
            self.vocab = Vocab(data_config_read, save_dir, self.train_filenames)
            self.vocab.update(self.dev_filenames)
        else:
            self.vocab = Vocab(data_config_read, save_dir)
            self.vocab.update(self.test_filenames)

        self.embedding_files = [embeddings_map['pretrained_embeddings'] for embeddings_map in
                                self.model_config['embeddings'].values()
                                if 'pretrained_embeddings' in embeddings_map]

        self.train_desc = train_desc
        # Generate mappings from feature/label names to indices in the model_fn inputs
        self.feature_idx_map = {}
        self.label_idx_map = {}
        for i, f in enumerate([d for d in self.data_config.keys() if
                               ('feature' in self.data_config[d] and self.data_config[d]['feature']) or
                               ('label' in self.data_config[d] and self.data_config[d]['label'])]):
            if 'feature' in self.data_config[f] and self.data_config[f]['feature']:
                self.feature_idx_map[f] = i
            if 'label' in self.data_config[f] and self.data_config[f]['label']:
                if 'type' in self.data_config[f] and self.data_config[f]['type'] == 'range':
                    idx = self.data_config[f]['conll_idx']
                    j = i + idx[1] if idx[1] != -1 else -1
                    self.label_idx_map[f] = (i, j)
                else:
                    self.label_idx_map[f] = (i, i + 1)

        print('feature_map:', self.feature_idx_map)
        print('label_map', self.label_idx_map)

        self.model = CJTMModel(self.hparams, self.model_config, self.data_config,
                               self.feature_idx_map, self.label_idx_map, self.vocab,
                               self.mode)

        # self._ops = self._gen_ops()

        self._save_vars = filter(lambda x: u'Pretrained' not in x.name, tf.global_variables())

        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'valid_loss': [],
            'valid_accuracy': [],
            'test_accuracy': 0
        }

    def train_input_fn(self, batch_size=None):
        if batch_size:
            return train_utils.get_input_fn(self.vocab, self.data_config, self.train_filenames, batch_size,
                                            num_epochs=self.hparams.num_train_epochs, shuffle=True,
                                            embedding_files=self.embedding_files,
                                            shuffle_buffer_multiplier=self.hparams.shuffle_buffer_multiplier)
        else:
            return train_utils.get_input_fn(self.vocab, self.data_config, self.train_filenames, self.hparams.batch_size,
                                            num_epochs=self.hparams.num_train_epochs, shuffle=True,
                                            embedding_files=self.embedding_files,
                                            shuffle_buffer_multiplier=self.hparams.shuffle_buffer_multiplier)

    def dev_input_fn(self):
        return train_utils.get_input_fn(self.vocab, self.data_config, self.dev_filenames,
                                        self.hparams.validate_batch_size,
                                        num_epochs=1, shuffle=False, embedding_files=self.embedding_files)

    def test_input_fn(self):
        return train_utils.get_input_fn(self.vocab, self.data_config, self.test_filenames,
                                        self.hparams.validate_batch_size,
                                        num_epochs=1, shuffle=False, embedding_files=self.embedding_files)

    @property
    def save_vars(self):
        return self._save_vars

    # def _gen_ops(self):
    #     outputs = self.model()

    def train(self, sess):
        print('********Training********')
        tf.logging.log(tf.logging.INFO, '********Training********')
        train_input_iterator = self.train_input_fn()
        train_input_next = train_input_iterator.get_next()
        dev_input_iterator = self.dev_input_fn()
        dev_input_next = dev_input_iterator.get_next()

        sess.run(train_input_iterator.initializer)
        sess.run(dev_input_iterator.initializer)
        train_input_iterators = {}
        train_input_next = {}
        dev_input_iterators = {}
        dev_input_next = {}
        self.train_history = {}
        self.val_history = {}

        # todo epoch and batch size should be different
        for k in self.train_desc:
            self.train_history[k] = {'step': [], 'acc': [], 'loss': [], 'f1': [], 'lr': float(self.model.hparams.learning_rate)}
            if k == 'srl':
                srl_batch_size = int(self.hparams.batch_size / 1)
                train_input_iterators[k] = self.train_input_fn(batch_size=srl_batch_size)
            else:
                train_input_iterators[k] = train_input_iterator
            # train_input_iterators[k] = self.train_input_fn()
            train_input_next[k] = train_input_iterators[k].get_next()
            # sess.run(train_input_iterators[k].initializer)
            self.val_history[k] = {'step': [], 'acc': [], 'loss': [], 'f1': [], 'best_eval': 0.0}
            dev_input_iterators[k] = dev_input_iterator
            # dev_input_iterators[k] = self.dev_input_fn()
            dev_input_next[k] = dev_input_iterators[k].get_next()
            # sess.run(dev_input_iterators[k].initializer)

        sess.run(tf.tables_initializer())
        steps = {}
        for key in self.train_desc:
            steps[key] = 0

        if os.path.isfile(os.path.join(self.save_dir, 'train_history.pkl')):
            train_history_file = open(os.path.join(self.save_dir, 'train_history.pkl'), 'rb')
            self.train_history = pickle.load(train_history_file)
            for key in self.train_desc:
                if self.train_history[key]['step']:
                    steps[key] = self.train_history[key]['step'][-1]
            train_history_file.close()
        if os.path.isfile(os.path.join(self.save_dir, 'val_history.pkl')):
            val_history_file = open(os.path.join(self.save_dir, 'val_history.pkl'), 'rb')
            self.val_history = pickle.load(val_history_file)
            val_history_file.close()




        train_start_time = time.time()
        # sys.stdout.flush()
        # save_path = os.path.join(self.save_dir, self.mode + 'CJMT')

        # print_every = 1
        # save_every = 1
        # validate_every = 1

        self.model.build_model(steps=steps)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=20)
        best_saver = tf.train.Saver(max_to_keep=20)
        checkpoint_files = os.listdir(self.save_dir)
        checkpoint_files = [x for x in checkpoint_files if x.endswith('_checkpoint')]
        start_at = 'pos'
        if checkpoint_files:
            checkpoint_files.sort(key=lambda fn: os.path.getmtime(os.path.join(self.save_dir, fn)))
            saver.restore(sess, tf.train.latest_checkpoint(self.save_dir, latest_filename=checkpoint_files[-1]))
            if checkpoint_files[-1].startswith('pos'):
                start_at = 'pos'
                # best_saver.restore(sess,
                #                    tf.train.latest_checkpoint(self.save_dir, latest_filename='pos_best_checkpoint'))
            elif checkpoint_files[-1].startswith('dep'):
                start_at = 'dep'
                # best_saver.restore(sess,
                #                    tf.train.latest_checkpoint(self.save_dir, latest_filename='dep_best_checkpoint'))
            else:
                start_at = 'srl'
                # best_saver.restore(sess,
                #                    tf.train.latest_checkpoint(self.save_dir, latest_filename='srl_best_checkpoint'))

        # if os.path.isfile(os.path.join(self.save_dir, 'srl_checkpoint')):
        #     saver.restore(sess, tf.train.latest_checkpoint(self.save_dir, latest_filename='srl_checkpoint'))
        # elif os.path.isfile(os.path.join(self.save_dir, 'dep_checkpoint')):
        #     saver.restore(sess, tf.train.latest_checkpoint(self.save_dir, latest_filename='dep_checkpoint'))
        # elif os.path.isfile(os.path.join(self.save_dir, 'pos_checkpoint')):
        #     saver.restore(sess, tf.train.latest_checkpoint(self.save_dir, latest_filename='pos_checkpoint'))

        # epoch
        lr_decay = 0.9

        for epoch in range(self.hparams.num_train_epochs):
            print_every = self.hparams.print_every
            validate_every = self.hparams.validate_every
            save_every = self.hparams.save_every
            if 'pos' in self.train_desc and start_at == 'pos':
                print('********Training POS layer********')
                tf.logging.log(tf.logging.INFO, '********Training POS layer********')
                # iter
                iter_pos = 1
                pos_lr_decay_count = 0
                self.model.hparams.set_hparam('learning_rate', self.train_history['pos']['lr'])
                while iter_pos < self.train_desc['pos'] + 1:
                    if iter_pos % (self.model.hparams.shuffle_buffer_multiplier * 5)\
                            == 0:
                        sess.run(train_input_iterators['pos'].initializer)
                    try:
                        pos_input = sess.run(train_input_next['pos'])
                        _, loss_pos, pos_steps, pos_lr, pos_predictions, pos_targets, tokens_to_keep = sess.run(
                            [self.model.optimize_op_pos, self.model.loss_pos,
                             self.model.pos_global_step, self.model.pos_lr, self.model.pos_predictions,
                             self.model.labels['gold_pos'], self.model.tokens_to_keep],
                            {self.model.features: pos_input})
                        if pos_steps % self.train_desc['pos']:
                            iter_pos = pos_steps % self.train_desc['pos']
                        else:
                            iter_pos = self.train_desc['pos']
                        self.train_history['pos']['step'].append(pos_steps)
                        self.train_history['pos']['loss'].append(loss_pos)
                        # batch_acc = 100.0 * tf.reduce_sum(
                        #     tf.to_float(tf.equal(pos_predictions, pos_targets)) * tokens_to_keep) / tf.reduce_sum(
                        #     tokens_to_keep)
                        # batch_acc = batch_acc.eval()
                        # tf.logging.log(tf.logging.INFO, "Train Steps:\t%d, pos_loss:\t%f\t, pos_learning_rate:\t%f, batch_pos_acc:\t%f%%"
                        #                % (pos_steps, loss_pos, pos_lr, batch_acc))
                        # print('pos_pred:', pos_predictions)
                        # print('pos_target:', pos_targets)

                        if iter_pos % print_every == 0:
                            print('pos_loss:', loss_pos)
                            tf.logging.log(tf.logging.INFO, "Train Steps:\t%d, pos_loss:\t%f\t, pos_learning_rate:\t%f"
                                           % (pos_steps, loss_pos, pos_lr))
                        if iter_pos % save_every == 0:
                            saver.save(sess, os.path.join(self.save_dir, self.mode + '_pos_pretrained_'),
                                       latest_filename='pos_checkpoint', global_step=pos_steps)
                            history_file = open(os.path.join(self.save_dir, 'train_history.pkl'), 'wb')
                            pickle.dump(self.train_history, history_file)
                            history_file.close()
                        if iter_pos and iter_pos % validate_every == 0:
                            num_count, num_correct, val_loss_list = 0, 0, []
                            self.model.mode = 'EVAL'
                            while True:
                                try:
                                    dev_pos_input = sess.run(dev_input_next['pos'])
                                    pos_predictions, pos_targets, tokens_to_keep, val_loss = sess.run(
                                        [self.model.pos_predictions, self.model.labels['gold_pos'],
                                         self.model.tokens_to_keep, self.model.loss_pos],
                                        {self.model.features: dev_pos_input})
                                    num_count += tf.reduce_sum(tokens_to_keep)
                                    val_loss_list.append(val_loss)
                                    batch_acc = 100.0 * tf.reduce_sum(tf.to_float(
                                        tf.equal(pos_predictions, pos_targets)) * tokens_to_keep) / tf.reduce_sum(
                                        tokens_to_keep)
                                    batch_acc = batch_acc.eval()
                                    tf.logging.log(tf.logging.INFO, 'batch accuracy:\t%f' % batch_acc)
                                    num_correct += tf.reduce_sum(
                                        tf.to_float(tf.equal(pos_predictions, pos_targets)) * tokens_to_keep)
                                except tf.errors.OutOfRangeError:
                                    break
                            sess.run(dev_input_iterators['pos'].initializer)
                            now_val_accuracy = (1.0 * num_correct / num_count * 100).eval()
                            now_val_loss = tf.reduce_mean(val_loss_list).eval()
                            if now_val_accuracy > self.val_history['pos']['best_eval']:
                                self.val_history['pos']['best_eval'] = now_val_accuracy
                                tf.logging.log(tf.logging.INFO,
                                               "Steps:\t%d, Validation_acc:\t%f%%, Validation_loss:\t%f\t*"
                                               % (pos_steps, now_val_accuracy, now_val_loss))
                                best_saver.save(sess, os.path.join(self.save_dir, 'pos_best_val_trained'),
                                                latest_filename='pos_best_checkpoint', global_step=pos_steps)
                            else:
                                tf.logging.log(tf.logging.INFO,
                                               "Steps:\t%d, Validation_acc:\t%f%%, Validation_loss:\t%f\t"
                                               % (pos_steps, now_val_accuracy, now_val_loss))
                                if iter_pos > validate_every and now_val_accuracy < self.val_history['pos']['acc'][-1]:
                                    pos_lr_decay_count += 1
                                else:
                                    pos_lr_decay_count = 0
                                if pos_lr_decay_count >= 10:
                                    self.train_history['pos']['lr'] = self.model.hparams.learning_rate * lr_decay
                                    self.model.hparams.set_hparam('learning_rate', self.train_history['pos']['lr'])
                                    pos_lr_decay_count = 0
                                    best_saver.restore(sess, tf.train.latest_checkpoint(self.save_dir,
                                                                                        latest_filename='pos_best_checkpoint'))
                                    continue

                            self.val_history['pos']['acc'].append(now_val_accuracy)
                            self.val_history['pos']['step'].append(pos_steps)
                            self.val_history['pos']['loss'].append(now_val_loss)
                            self.model.mode = 'TRAIN'
                            history_file = open(os.path.join(self.save_dir, 'val_history.pkl'), 'wb')
                            pickle.dump(self.val_history, history_file)
                            history_file.close()
                        iter_pos += 1
                    except tf.errors.InvalidArgumentError:
                        # best_saver.restore(sess, tf.train.latest_checkpoint(self.save_dir,
                        #                                                     latest_filename='pos_best_checkpoint'))
                        saver.restore(sess, tf.train.latest_checkpoint(self.save_dir, latest_filename='pos_checkpoint'))

                    except tf.errors.ResourceExhaustedError:
                        saver.restore(sess, tf.train.latest_checkpoint(self.save_dir, latest_filename='pos_checkpoint'))

                start_at = 'dep'

            if 'dep' in self.train_desc and start_at == 'dep':
                print('********Training DEP layer********')
                tf.logging.log(tf.logging.INFO, '********Training DEP layer********')
                print_every = self.hparams.print_every / 2
                validate_every = self.hparams.validate_every
                # validate_every = 100
                save_every = self.hparams.save_every
                iter_dep = 1
                dep_lr_decay_count = 0
                self.model.hparams.set_hparam('learning_rate', self.train_history['dep']['lr'])
                while iter_dep < self.train_desc['dep'] + 1:
                    if iter_dep % (self.model.hparams.shuffle_buffer_multiplier * 5)\
                            == 0:
                        sess.run(train_input_iterators['dep'].initializer)
                    try:
                        dep_input = sess.run(train_input_next['dep'])
                        _, loss_dep, dep_steps, dep_lr = sess.run([self.model.optimize_op_dep, self.model.loss_dep,
                                                                   self.model.dep_global_step, self.model.dep_lr],
                                                                  {self.model.features: dep_input})
                        if dep_steps % self.train_desc['dep']:
                            iter_dep = dep_steps % self.train_desc['dep']
                        else:
                            iter_dep = self.train_desc['dep']
                        self.train_history['dep']['step'].append(dep_steps)
                        self.train_history['dep']['loss'].append(loss_dep)
                        if iter_dep % print_every == 0:
                            print('dep_loss:', loss_dep)
                            tf.logging.log(tf.logging.INFO, "Train Steps:\t%d, dep_loss:\t%f, dep_learning_rate:\t%f"
                                           % (dep_steps, loss_dep, dep_lr))
                        if iter_dep % save_every == 0:
                            saver.save(sess, os.path.join(self.save_dir, self.mode + '_dep_pretrained_'),
                                       latest_filename='dep_checkpoint', global_step=dep_steps)
                            history_file = open(os.path.join(self.save_dir, 'train_history.pkl'), 'wb')
                            pickle.dump(self.train_history, history_file)
                            history_file.close()
                        if iter_dep and iter_dep % validate_every == 0:
                            num_count, num_head_correct, val_loss_list = 0, 0, []
                            now_dep_accuracies = tf.constant([0, 0, 0], dtype=tf.int64)
                            now_total = tf.constant(0, dtype=tf.int64)
                            self.model.mode = 'EVAL'

                            while True:
                                try:
                                    dev_dep_input = sess.run(dev_input_next['dep'])
                                    rel_predictions, rel_targets, head_predictions, head_targets, tokens_to_keep, val_loss, \
                                    words, pos_targets = sess.run(
                                        [self.model.rel_predictions, self.model.labels['parse_label'],
                                         self.model.head_predictions, self.model.labels['parse_head'],
                                         self.model.tokens_to_keep, self.model.loss_dep,
                                         self.model.feats['word'], self.model.labels['gold_pos']],
                                        {self.model.features: dev_dep_input})

                                    total, dep_accuracies = evaluates.conll_parse_eval(predictions=rel_predictions,
                                                                                       targets=rel_targets,
                                                                                       parse_head_predictions=head_predictions,
                                                                                       parse_head_targets=head_targets,
                                                                                       words=words,
                                                                                       mask=tokens_to_keep,
                                                                                       reverse_maps=self.vocab.reverse_maps,
                                                                                       gold_parse_eval_file=os.path.join(
                                                                                           self.save_dir,
                                                                                           'parse_gold.txt'),
                                                                                       pred_parse_eval_file=os.path.join(
                                                                                           self.save_dir,
                                                                                           'parse_pred.txt'),
                                                                                       pos_targets=pos_targets)
                                    now_dep_accuracies += dep_accuracies
                                    now_total += total
                                    num_count += tf.reduce_sum(tokens_to_keep)
                                    val_loss_list.append(val_loss)
                                    num_head_correct += tf.reduce_sum(
                                        tf.to_float(tf.equal(head_predictions, head_targets)) * tokens_to_keep)
                                except tf.errors.OutOfRangeError:
                                    break

                            sess.run(dev_input_iterators['dep'].initializer)
                            now_val_head_accuracy = (1.0 * num_head_correct / num_count * 100).eval()
                            now_val_loss = tf.reduce_mean(val_loss_list).eval()
                            now_dep_accuracies = tf.to_float(now_dep_accuracies) / tf.to_float(now_total) * 100
                            now_dep_accuracies = now_dep_accuracies.eval()
                            if now_dep_accuracies[-1] > self.val_history['dep']['best_eval']:
                                self.val_history['dep']['best_eval'] = now_dep_accuracies[-1]
                                tf.logging.log(tf.logging.INFO, "Steps:\t%d, Validation_head_acc:\t%f%%,"
                                                                "Validation_UAS:\t%f%%, Validation_LAS:\t%f%%,"
                                                                " Validation_label_acc:\t%f%% , Validation_loss:\t%f\t*"
                                               % (dep_steps, now_val_head_accuracy, now_dep_accuracies[0],
                                                  now_dep_accuracies[1], now_dep_accuracies[2], now_val_loss))
                                best_saver.save(sess, os.path.join(self.save_dir, 'dep_best_val_trained'),
                                                latest_filename='dep_best_checkpoint', global_step=dep_steps)
                            else:
                                tf.logging.log(tf.logging.INFO, "Steps:\t%d, Validation_head_acc:\t%f%%,"
                                                                "Validation_LAS:\t%f%%, Validation_UAS:\t%f%%,"
                                                                " Validation_label_acc:\t%f%% , Validation_loss:\t%f\t"
                                               % (dep_steps, now_val_head_accuracy, now_dep_accuracies[0],
                                                  now_dep_accuracies[1], now_dep_accuracies[2], now_val_loss))
                                if iter_dep > validate_every and now_dep_accuracies[1] < \
                                        self.val_history['dep']['acc'][-1]['UAS']:
                                    dep_lr_decay_count += 1
                                else:
                                    dep_lr_decay_count = 0
                                if dep_lr_decay_count >= 10:
                                    self.train_history['dep']['lr'] = self.model.hparams.learning_rate * lr_decay
                                    self.model.hparams.set_hparam('learning_rate',
                                                                 self.train_history['dep']['lr'])
                                    dep_lr_decay_count = 0
                                    best_saver.restore(sess, tf.train.latest_checkpoint(self.save_dir,
                                                                                        latest_filename='dep_best_checkpoint'))
                                    continue

                            self.val_history['dep']['acc'].append({'head_acc': now_val_head_accuracy,
                                                                   'LAS': now_dep_accuracies[0],
                                                                   'UAS': now_dep_accuracies[1],
                                                                   'label_acc': now_dep_accuracies[2]})
                            self.val_history['dep']['step'].append(dep_steps)
                            self.val_history['dep']['loss'].append(now_val_loss)
                            self.model.mode = 'TRAIN'
                            history_file = open(os.path.join(self.save_dir, 'val_history.pkl'), 'wb')
                            pickle.dump(self.val_history, history_file)
                            history_file.close()
                        iter_dep += 1
                    except tf.errors.InvalidArgumentError:
                        # best_saver.restore(sess, tf.train.latest_checkpoint(self.save_dir,
                        #                                                     latest_filename='dep_best_checkpoint'))
                        saver.restore(sess, tf.train.latest_checkpoint(self.save_dir, latest_filename='dep_checkpoint'))

                    except tf.errors.ResourceExhaustedError:
                        saver.restore(sess, tf.train.latest_checkpoint(self.save_dir, latest_filename='dep_checkpoint'))

                start_at = 'srl'

            if 'srl' in self.train_desc and start_at == 'srl':
                print_every = self.hparams.print_every
                validate_every = self.hparams.validate_every
                save_every = self.hparams.save_every
                print('********Training SRL layer********')
                tf.logging.log(tf.logging.INFO, '********Training SRL layer********')
                #
                iter_srl = 1
                srl_lr_decay_count = 0
                self.model.hparams.set_hparam('learning_rate', self.train_history['srl']['lr'])
                while iter_srl < self.train_desc['srl'] + 1:
                    if iter_srl % (self.model.hparams.shuffle_buffer_multiplier * 5)\
                            == 0:
                        sess.run(train_input_iterators['srl'].initializer)
                    try:
                        srl_input = sess.run(train_input_next['srl'])
                        _, _, loss_srl, loss_pre, srl_steps, srl_lr = sess.run(
                            [self.model.optimize_op_srl, self.model.optimize_op_pre,
                             self.model.loss_srl, self.model.loss_pre,
                             self.model.srl_global_step, self.model.srl_lr],
                            {self.model.features: srl_input})
                        if srl_steps % self.train_desc['srl']:
                            iter_srl = srl_steps % self.train_desc['srl']
                        else:
                            iter_srl = self.train_desc['srl']

                        self.train_history['srl']['step'].append(srl_steps)
                        self.train_history['srl']['loss'].append({'srl': loss_srl, 'predicate': loss_pre})
                        if iter_srl % print_every == 0:
                            print('srl_loss:', loss_srl)
                            print('pred_loss', loss_pre)
                            tf.logging.log(tf.logging.INFO,
                                           "Train Steps:\t%d, srl_loss:\t%f, pred_loss:\t%f, srl_learning_rate:\t%f"
                                           % (srl_steps, loss_srl, loss_pre, srl_lr))
                        if iter_srl % save_every == 0:
                            saver.save(sess, os.path.join(self.save_dir, self.mode + '_srl_pretrained_'),
                                       latest_filename='srl_checkpoint', global_step=srl_steps)
                            history_file = open(os.path.join(self.save_dir, 'train_history.pkl'), 'wb')
                            pickle.dump(self.train_history, history_file)
                            history_file.close()
                        if iter_srl and iter_srl % validate_every == 0:
                            self.model.mode = 'EVAL'
                            num_count, num_pred_correct, val_loss_list, val_pred_loss_list = 0, 0, [], []
                            now_srl_correct = tf.constant(0, dtype=tf.int64)
                            now_srl_missed = tf.constant(0, dtype=tf.int64)
                            now_srl_excess = tf.constant(0, dtype=tf.int64)

                            while True:
                                try:
                                    dev_srl_input = sess.run(dev_input_next['srl'])
                                    rel_predictions, rel_targets, head_predictions, head_targets, tokens_to_keep, val_loss, \
                                    words, pos_targets, pos_predictions, srl_predictions, srl_targets, predicate_predictions, \
                                    predicate_targets, pred_loss = sess.run(
                                        [self.model.rel_predictions, self.model.labels['parse_label'],
                                         self.model.head_predictions, self.model.labels['parse_head'],
                                         self.model.tokens_to_keep, self.model.loss_srl,
                                         self.model.feats['word'], self.model.labels['gold_pos'],
                                         self.model.pos_predictions,
                                         self.model.srl_predictions, self.model.srl_targets, self.model.pre_predictions,
                                         self.model.labels['predicate'], self.model.loss_pre],
                                        {self.model.features: dev_srl_input})
                                    correct, excess, missed = evaluates.conll_srl_eval(predictions=srl_predictions,
                                                                                       targets=srl_targets,
                                                                                       predicate_targets=predicate_targets,
                                                                                       predicate_predictions=predicate_predictions,
                                                                                       parse_head_targets=head_targets,
                                                                                       parse_head_predictions=head_predictions,
                                                                                       parse_label_targets=rel_targets,
                                                                                       parse_label_predictions=rel_predictions,
                                                                                       words=words,
                                                                                       reverse_maps=self.vocab.reverse_maps,
                                                                                       gold_srl_eval_file=os.path.join(
                                                                                           self.save_dir,
                                                                                           'srl_gold.txt'),
                                                                                       pred_srl_eval_file=os.path.join(
                                                                                           self.save_dir,
                                                                                           'srl_pred.txt'),
                                                                                       mask=tokens_to_keep,
                                                                                       pos_predictions=pos_predictions,
                                                                                       pos_targets=pos_targets)
                                    now_srl_correct += correct
                                    now_srl_missed += missed
                                    now_srl_excess += excess
                                    num_count += tf.reduce_sum(tokens_to_keep)
                                    val_loss_list.append(val_loss)
                                    val_pred_loss_list.append(pred_loss)
                                    num_pred_correct += tf.reduce_sum(
                                        tf.to_float(
                                            tf.equal(predicate_predictions, predicate_targets)) * tokens_to_keep)
                                except tf.errors.OutOfRangeError:
                                    break

                            sess.run(dev_input_iterators['srl'].initializer)
                            now_val_pred_accuracy = (1.0 * num_pred_correct / num_count * 100).eval()
                            now_val_loss = tf.reduce_mean(val_loss_list).eval()
                            now_pred_loss = tf.reduce_mean(val_pred_loss_list).eval()
                            precision = tf.to_float(now_srl_correct) / tf.to_float(now_srl_correct + now_srl_excess) * 100
                            recall = tf.to_float(now_srl_correct) / tf.to_float(now_srl_correct + now_srl_missed) * 100
                            f1 = 2 * precision * recall / (precision + recall)
                            now_srl_f1 = f1.eval()

                            if now_srl_f1 > self.val_history['srl']['best_eval']:
                                self.val_history['srl']['best_eval'] = now_srl_f1
                                tf.logging.log(tf.logging.INFO, "Steps:\t%d, Validation_pred_acc:\t%f%%,"
                                                                " Validation_pred_loss:\t%f, "
                                                                "Validation_f1:\t%f%%, Validation_srl_loss:\t%f\t*"
                                               % (
                                                   srl_steps, now_val_pred_accuracy, now_pred_loss, now_srl_f1,
                                                   now_val_loss))
                                best_saver.save(sess, os.path.join(self.save_dir, 'srl_best_val_trained'),
                                                latest_filename='srl_best_checkpoint', global_step=srl_steps)
                            else:
                                tf.logging.log(tf.logging.INFO, "Steps:\t%d, Validation_pred_acc:\t%f%%,"
                                                                " Validation_pred_loss:\t%f, "
                                                                "Validation_f1:\t%f%%, Validation_srl_loss:\t%f\t"
                                               % (
                                                   srl_steps, now_val_pred_accuracy, now_pred_loss, now_srl_f1,
                                                   now_val_loss))
                                if iter_srl > validate_every and now_srl_f1 < self.val_history['srl']['f1'][-1]:
                                    srl_lr_decay_count += 1
                                else:
                                    srl_lr_decay_count = 0
                                if srl_lr_decay_count >= 10:
                                    self.train_history['srl']['lr'] = self.model.hparams.learning_rate * lr_decay
                                    self.model.hparams.set_hparam('learning_rate',
                                                                 self.train_history['srl']['lr'])
                                    srl_lr_decay_count = 0
                                    best_saver.restore(sess, tf.train.latest_checkpoint(self.save_dir,
                                                                                        latest_filename='srl_best_checkpoint'))
                                    continue

                            self.val_history['srl']['acc'].append(now_val_pred_accuracy)
                            self.val_history['srl']['step'].append(srl_steps)
                            self.val_history['srl']['loss'].append({'srl': now_val_loss, 'predicate': now_pred_loss})
                            self.val_history['srl']['f1'].append(now_srl_f1)
                            self.model.mode = 'TRAIN'
                            history_file = open(os.path.join(self.save_dir, 'val_history.pkl'), 'wb')
                            pickle.dump(self.val_history, history_file)
                            history_file.close()
                        iter_srl += 1
                    except tf.errors.InvalidArgumentError:
                        # best_saver.restore(sess, tf.train.latest_checkpoint(self.save_dir,
                        #                                                     latest_filename='srl_best_checkpoint'))
                        saver.restore(sess, tf.train.latest_checkpoint(self.save_dir, latest_filename='srl_checkpoint'))

                    except tf.errors.ResourceExhaustedError:
                        saver.restore(sess, tf.train.latest_checkpoint(self.save_dir, latest_filename='srl_checkpoint'))

                start_at = 'pos'

    def test(self, sess):
        print('********Testing********')
        tf.logging.log(tf.logging.INFO, '********Testing********')
        dev_input_iterator = self.dev_input_fn()
        dev_input_next = dev_input_iterator.get_next()
        test_input_iterator = self.test_input_fn()
        test_input_next = test_input_iterator.get_next()
        sess.run(test_input_iterator.initializer)
        sess.run(dev_input_iterator.initializer)
        test_input_iterators = {}
        test_input_next = {}
        dev_input_iterators = {}
        dev_input_next = {}

        for k in self.train_desc:
            test_input_iterators[k] = test_input_iterator
            test_input_next[k] = test_input_iterators[k].get_next()
            dev_input_iterators[k] = dev_input_iterator
            dev_input_next[k] = dev_input_iterators[k].get_next()

        sess.run(tf.tables_initializer())
        steps = {}
        for key in self.train_desc:
            steps[key] = 0

        self.model.build_model(steps=steps)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=20)
        checkpoint_files = os.listdir(self.save_dir)
        checkpoint_files = [x for x in checkpoint_files if x.endswith('_checkpoint')]
        # if checkpoint_files:
        #     checkpoint_files.sort(key=lambda fn: os.path.getmtime(os.path.join(self.save_dir, fn)))
        #     saver.restore(sess, tf.test.latest_checkpoint(self.save_dir, latest_filename=checkpoint_files[-1]))

        # if os.path.isfile(os.path.join(self.save_dir, 'srl_checkpoint')):
        #     saver.restore(sess, tf.test.latest_checkpoint(self.save_dir, latest_filename='srl_checkpoint'))
        # elif os.path.isfile(os.path.join(self.save_dir, 'dep_checkpoint')):
        #     saver.restore(sess, tf.test.latest_checkpoint(self.save_dir, latest_filename='dep_checkpoint'))
        # elif os.path.isfile(os.path.join(self.save_dir, 'pos_checkpoint')):
        #     saver.restore(sess, tf.test.latest_checkpoint(self.save_dir, latest_filename='pos_checkpoint'))

        # todo test_desc
        if 'pos' in self.train_desc:
            print('********Testing POS layer********')
            tf.logging.log(tf.logging.INFO, '********Testing POS layer********')
            saver.restore(sess, tf.train.latest_checkpoint(self.save_dir, latest_filename='pos_best_checkpoint'))
            num_count, num_correct, val_loss_list = 0, 0, []
            # self.model.mode = 'EVAL'
            while True:
                try:
                    dev_pos_input = sess.run(dev_input_next['pos'])
                    pos_predictions, pos_targets, tokens_to_keep, val_loss = sess.run(
                        [self.model.pos_predictions, self.model.labels['gold_pos'],
                         self.model.tokens_to_keep, self.model.loss_pos],
                        {self.model.features: dev_pos_input})
                    num_count += tf.reduce_sum(tokens_to_keep)
                    val_loss_list.append(val_loss)
                    num_correct += tf.reduce_sum(
                        tf.to_float(tf.equal(pos_predictions, pos_targets)) * tokens_to_keep)
                except tf.errors.OutOfRangeError:
                    break
            sess.run(dev_input_iterators['pos'].initializer)
            now_val_accuracy = (1.0 * num_correct / num_count * 100).eval()
            now_val_loss = tf.reduce_mean(val_loss_list).eval()
            tf.logging.log(tf.logging.INFO, "Validation_acc:\t%f%%, Validation_loss:\t%f\t"
                           % (now_val_accuracy, now_val_loss))

            num_count, num_correct, val_loss_list = 0, 0, []
            self.model.mode = 'EVAL'
            while True:
                try:
                    test_pos_input = sess.run(test_input_next['pos'])
                    pos_predictions, pos_targets, tokens_to_keep, val_loss = sess.run(
                        [self.model.pos_predictions, self.model.labels['gold_pos'],
                         self.model.tokens_to_keep, self.model.loss_pos],
                        {self.model.features: test_pos_input})
                    num_count += tf.reduce_sum(tokens_to_keep)
                    val_loss_list.append(val_loss)
                    num_correct += tf.reduce_sum(
                        tf.to_float(tf.equal(pos_predictions, pos_targets)) * tokens_to_keep)
                except tf.errors.OutOfRangeError:
                    break
            sess.run(test_input_iterators['pos'].initializer)
            now_val_accuracy = (1.0 * num_correct / num_count * 100).eval()
            now_val_loss = tf.reduce_mean(val_loss_list).eval()
            tf.logging.log(tf.logging.INFO, "Test_acc:\t%f%%, Test_loss:\t%f"
                           % (now_val_accuracy, now_val_loss))

        if 'dep' in self.train_desc:
            print('********Testing DEP layer********')
            tf.logging.log(tf.logging.INFO, '********Testing DEP layer********')
            num_count, num_head_correct, val_loss_list = 0, 0, []

            now_dep_accuracies = tf.constant([0, 0, 0], dtype=tf.int64)
            now_total = tf.constant(0, dtype=tf.int64)
            # self.model.mode = 'EVAL'
            saver.restore(sess, tf.train.latest_checkpoint(self.save_dir, latest_filename='dep_best_checkpoint'))

            while True:
                try:
                    dev_dep_input = sess.run(dev_input_next['dep'])
                    rel_predictions, rel_targets, head_predictions, head_targets, tokens_to_keep, val_loss, \
                    words, pos_targets = sess.run(
                        [self.model.rel_predictions, self.model.labels['parse_label'],
                         self.model.head_predictions, self.model.labels['parse_head'],
                         self.model.tokens_to_keep, self.model.loss_dep,
                         self.model.feats['word'], self.model.labels['gold_pos']],
                        {self.model.features: dev_dep_input})

                    total, dep_accuracies = evaluates.conll_parse_eval(predictions=rel_predictions,
                                                                       targets=rel_targets,
                                                                       parse_head_predictions=head_predictions,
                                                                       parse_head_targets=head_targets,
                                                                       words=words,
                                                                       mask=tokens_to_keep,
                                                                       reverse_maps=self.vocab.reverse_maps,
                                                                       gold_parse_eval_file=os.path.join(
                                                                           self.save_dir,
                                                                           'parse_gold.txt'),
                                                                       pred_parse_eval_file=os.path.join(
                                                                           self.save_dir,
                                                                           'parse_pred.txt'),
                                                                       pos_targets=pos_targets)
                    now_dep_accuracies += dep_accuracies
                    now_total += total
                    num_count += tf.reduce_sum(tokens_to_keep)
                    val_loss_list.append(val_loss)
                    num_head_correct += tf.reduce_sum(
                        tf.to_float(tf.equal(head_predictions, head_targets)) * tokens_to_keep)
                except tf.errors.OutOfRangeError:
                    break

            sess.run(dev_input_iterators['dep'].initializer)
            now_val_head_accuracy = (1.0 * num_head_correct / num_count * 100).eval()
            now_val_loss = tf.reduce_mean(val_loss_list).eval()
            now_dep_accuracies = tf.to_float(now_dep_accuracies) / tf.to_float(now_total) * 100
            now_dep_accuracies = now_dep_accuracies.eval()
            tf.logging.log(tf.logging.INFO, "Validation_head_acc:\t%f%%,"
                                            "Validation_LAS:\t%f%%, Validation_UAS:\t%f%%,"
                                            " Validation_label_acc:\t%f%% , Validation_loss:\t%f"
                           % (now_val_head_accuracy, now_dep_accuracies[0],
                              now_dep_accuracies[1], now_dep_accuracies[2], now_val_loss))

            num_count, num_head_correct, val_loss_list = 0, 0, []
            now_dep_accuracies = tf.constant([0, 0, 0], dtype=tf.int64)
            now_total = tf.constant(0, dtype=tf.int64)
            while True:
                try:
                    test_dep_input = sess.run(test_input_next['dep'])
                    rel_predictions, rel_targets, head_predictions, head_targets, tokens_to_keep, val_loss, \
                    words, pos_targets = sess.run(
                        [self.model.rel_predictions, self.model.labels['parse_label'],
                         self.model.head_predictions, self.model.labels['parse_head'],
                         self.model.tokens_to_keep, self.model.loss_dep,
                         self.model.feats['word'], self.model.labels['gold_pos']],
                        {self.model.features: test_dep_input})

                    total, dep_accuracies = evaluates.conll_parse_eval(predictions=rel_predictions,
                                                                       targets=rel_targets,
                                                                       parse_head_predictions=head_predictions,
                                                                       parse_head_targets=head_targets,
                                                                       words=words,
                                                                       mask=tokens_to_keep,
                                                                       reverse_maps=self.vocab.reverse_maps,
                                                                       gold_parse_eval_file=os.path.join(
                                                                           self.save_dir,
                                                                           'parse_gold.txt'),
                                                                       pred_parse_eval_file=os.path.join(
                                                                           self.save_dir,
                                                                           'parse_pred.txt'),
                                                                       pos_targets=pos_targets)
                    now_dep_accuracies += dep_accuracies
                    now_total += total
                    num_count += tf.reduce_sum(tokens_to_keep)
                    val_loss_list.append(val_loss)
                    num_head_correct += tf.reduce_sum(
                        tf.to_float(tf.equal(head_predictions, head_targets)) * tokens_to_keep)
                except tf.errors.OutOfRangeError:
                    break

            sess.run(test_input_iterators['dep'].initializer)
            now_val_head_accuracy = (1.0 * num_head_correct / num_count * 100).eval()
            now_val_loss = tf.reduce_mean(val_loss_list).eval()
            now_dep_accuracies = tf.to_float(now_dep_accuracies) / tf.to_float(now_total) * 100
            now_dep_accuracies = now_dep_accuracies.eval()
            tf.logging.log(tf.logging.INFO, "Test_head_acc:\t%f%%,"
                                            "Test_LAS:\t%f%%, Test_UAS:\t%f%%,"
                                            "Test_label_acc:\t%f%% , Test_loss:\t%f"
                           % (now_val_head_accuracy, now_dep_accuracies[0],
                              now_dep_accuracies[1], now_dep_accuracies[2], now_val_loss))

        if 'srl' in self.train_desc:
            print('********Testing SRL layer********')
            tf.logging.log(tf.logging.INFO, '********Testing SRL layer********')
            # self.model.mode = 'EVAL'
            saver.restore(sess, tf.train.latest_checkpoint(self.save_dir, latest_filename='srl_best_checkpoint'))

            num_count, num_pred_correct, val_loss_list, val_pred_loss_list = 0, 0, [], []
            now_srl_correct = tf.constant(0, dtype=tf.int64)
            now_srl_missed = tf.constant(0, dtype=tf.int64)
            now_srl_excess = tf.constant(0, dtype=tf.int64)

            while True:
                try:
                    dev_srl_input = sess.run(dev_input_next['srl'])
                    rel_predictions, rel_targets, head_predictions, head_targets, tokens_to_keep, val_loss, \
                    words, pos_targets, pos_predictions, srl_predictions, srl_targets, predicate_predictions, \
                    predicate_targets, pred_loss = sess.run(
                        [self.model.rel_predictions, self.model.labels['parse_label'],
                         self.model.head_predictions, self.model.labels['parse_head'],
                         self.model.tokens_to_keep, self.model.loss_srl,
                         self.model.feats['word'], self.model.labels['gold_pos'],
                         self.model.pos_predictions,
                         self.model.srl_predictions, self.model.srl_targets, self.model.pre_predictions,
                         self.model.labels['predicate'], self.model.loss_pre],
                        {self.model.features: dev_srl_input})
                    correct, excess, missed = evaluates.conll_srl_eval(predictions=srl_predictions,
                                                                       targets=srl_targets,
                                                                       predicate_targets=predicate_targets,
                                                                       predicate_predictions=predicate_predictions,
                                                                       parse_head_targets=head_targets,
                                                                       parse_head_predictions=head_predictions,
                                                                       parse_label_targets=rel_targets,
                                                                       parse_label_predictions=rel_predictions,
                                                                       words=words,
                                                                       reverse_maps=self.vocab.reverse_maps,
                                                                       gold_srl_eval_file=os.path.join(
                                                                           self.save_dir,
                                                                           'srl_gold.txt'),
                                                                       pred_srl_eval_file=os.path.join(
                                                                           self.save_dir,
                                                                           'srl_pred.txt'),
                                                                       mask=tokens_to_keep,
                                                                       pos_predictions=pos_predictions,
                                                                       pos_targets=pos_targets)
                    now_srl_correct += correct
                    now_srl_missed += missed
                    now_srl_excess += excess
                    num_count += tf.reduce_sum(tokens_to_keep)
                    val_loss_list.append(val_loss)
                    val_pred_loss_list.append(pred_loss)
                    num_pred_correct += tf.reduce_sum(
                        tf.to_float(tf.equal(predicate_predictions, predicate_targets)) * tokens_to_keep)
                except tf.errors.OutOfRangeError:
                    break

            sess.run(dev_input_iterators['srl'].initializer)
            now_val_pred_accuracy = (1.0 * num_pred_correct / num_count * 100).eval()
            now_val_loss = tf.reduce_mean(val_loss_list).eval()
            now_pred_loss = tf.reduce_mean(val_pred_loss_list).eval()
            precision = tf.to_float(now_srl_correct) / tf.to_float(now_srl_correct + now_srl_excess) * 100
            recall = tf.to_float(now_srl_correct) / tf.to_float(now_srl_correct + now_srl_missed) * 100
            f1 = 2 * precision * recall / (precision + recall)
            now_srl_f1 = f1.eval()

            tf.logging.log(tf.logging.INFO, "Validation_pred_acc:\t%f%%,"
                                            "Validation_pred_loss:\t%f, "
                                            "Validation_f1:\t%f%%, Validation_srl_loss:\t%f\t"
                           % (now_val_pred_accuracy, now_pred_loss, now_srl_f1,
                              now_val_loss))

            num_count, num_pred_correct, val_loss_list, val_pred_loss_list = 0, 0, [], []
            now_srl_correct = tf.constant(0, dtype=tf.int64)
            now_srl_missed = tf.constant(0, dtype=tf.int64)
            now_srl_excess = tf.constant(0, dtype=tf.int64)

            while True:
                try:
                    test_srl_input = sess.run(test_input_next['srl'])
                    rel_predictions, rel_targets, head_predictions, head_targets, tokens_to_keep, val_loss, \
                    words, pos_targets, pos_predictions, srl_predictions, srl_targets, predicate_predictions, \
                    predicate_targets, pred_loss = sess.run(
                        [self.model.rel_predictions, self.model.labels['parse_label'],
                         self.model.head_predictions, self.model.labels['parse_head'],
                         self.model.tokens_to_keep, self.model.loss_srl,
                         self.model.feats['word'], self.model.labels['gold_pos'],
                         self.model.pos_predictions,
                         self.model.srl_predictions, self.model.srl_targets, self.model.pre_predictions,
                         self.model.labels['predicate'], self.model.loss_pre],
                        {self.model.features: test_srl_input})
                    correct, excess, missed = evaluates.conll_srl_eval(predictions=srl_predictions,
                                                                       targets=srl_targets,
                                                                       predicate_targets=predicate_targets,
                                                                       predicate_predictions=predicate_predictions,
                                                                       parse_head_targets=head_targets,
                                                                       parse_head_predictions=head_predictions,
                                                                       parse_label_targets=rel_targets,
                                                                       parse_label_predictions=rel_predictions,
                                                                       words=words,
                                                                       reverse_maps=self.vocab.reverse_maps,
                                                                       gold_srl_eval_file=os.path.join(
                                                                           self.save_dir,
                                                                           'srl_gold.txt'),
                                                                       pred_srl_eval_file=os.path.join(
                                                                           self.save_dir,
                                                                           'srl_pred.txt'),
                                                                       mask=tokens_to_keep,
                                                                       pos_predictions=pos_predictions,
                                                                       pos_targets=pos_targets)
                    now_srl_correct += correct
                    now_srl_missed += missed
                    now_srl_excess += excess
                    num_count += tf.reduce_sum(tokens_to_keep)
                    val_loss_list.append(val_loss)
                    val_pred_loss_list.append(pred_loss)
                    num_pred_correct += tf.reduce_sum(
                        tf.to_float(tf.equal(predicate_predictions, predicate_targets)) * tokens_to_keep)
                except tf.errors.OutOfRangeError:
                    break

            sess.run(test_input_iterators['srl'].initializer)
            now_val_pred_accuracy = (1.0 * num_pred_correct / num_count * 100).eval()
            now_val_loss = tf.reduce_mean(val_loss_list).eval()
            now_pred_loss = tf.reduce_mean(val_pred_loss_list).eval()
            precision = tf.to_float(now_srl_correct) / tf.to_float(now_srl_correct + now_srl_excess) * 100
            recall = tf.to_float(now_srl_correct) / tf.to_float(now_srl_correct + now_srl_missed) * 100
            f1 = 2 * precision * recall / (precision + recall)
            now_srl_f1 = f1.eval()

            tf.logging.log(tf.logging.INFO, "Test_pred_acc:\t%f%%,"
                                            "Test_pred_loss:\t%f, "
                                            "Test_f1:\t%f%%, Test_srl_loss:\t%f\t"
                           % (now_val_pred_accuracy, now_pred_loss, now_srl_f1,
                              now_val_loss))
