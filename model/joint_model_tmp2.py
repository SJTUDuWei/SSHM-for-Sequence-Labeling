# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2018/12/15 下午5:30
# @Project: unified_chinese_multi_tasking_framework
# @File: joint_model.py
# @Software: PyCharm

# 参考LISA-V1  加入dependency parsing srl multi-head attention biaffie(bilinear classification)
# 修改数据格式 todo add character input list in dataset.py
import numpy as np
import tensorflow as tf
import cws_pos
import dependency_parsing
import srl as srl_op
# from lang_model import CharacterLanguageModel
# todo character embedding => CNN => word representation
import nn_utils
import transformer

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

    @staticmethod
    def load_transitions(transition_statistics, num_classes, vocab_map):
        transition_statistics_np = np.zeros((num_classes, num_classes))
        with open(transition_statistics, 'r') as f:
            for line in f:
                tag1, tag2, prob = line.split("\t")
                transition_statistics_np[vocab_map[tag1], vocab_map[tag2]] = float(prob)
        return transition_statistics_np

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

    def build_model(self, steps):
        ''' Builds the whole computational graph '''
        self.pos_global_step = tf.Variable(initial_value=steps['pos'], trainable=False, name='pos_global_step')
        self.dep_global_step = tf.Variable(initial_value=steps['dep'], trainable=False, name='dep_global_step')
        self.pre_global_step = tf.Variable(initial_value=steps['srl'], trainable=False, name='pre_global_step')
        self.srl_global_step = tf.Variable(initial_value=steps['srl'], trainable=False, name='srl_global_step')

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
                these_labels_masked = tf.multiply(these_labels,
                                                  tf.cast(tf.expand_dims(self.tokens_to_keep, -1), tf.int32))
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

            self.optimize_op_pos, concat_outputs_pos, self.loss_pos, \
            self.pos_predictions, self.pos_scores, self.pos_probabilities, \
            self.pos_num_heads, self.pos_lr = cws_pos.pos_op(embeds=current_input,
                                                             tokens_to_keep=self.tokens_to_keep,
                                                             model_config=self.model_config,
                                                             hparams=self.hparams, vocab=self.vocab,
                                                             t_pos=labels['gold_pos'],
                                                             pos_global_step=self.pos_global_step)

            self.optimize_op_dep, concat_outputs_dep, self.loss_dep, \
            self.head_predictions, self.head_scores, self.head_probabilites, \
            self.rel_predictions, self.rel_scores, self.rel_probabilities, \
            self.dep_num_heads, self.dep_lr = dependency_parsing.dep_op(concat_outputs_pos=concat_outputs_pos,
                                                                        tokens_to_keep=self.tokens_to_keep,
                                                                        model_config=self.model_config,
                                                                        hparams=self.hparams, vocab=self.vocab,
                                                                        t_head=labels['parse_head'],
                                                                        t_dep=labels['parse_label'],
                                                                        mode=self.mode,
                                                                        pos_num_heads=self.pos_num_heads,
                                                                        pos_probabilities=self.pos_probabilities,
                                                                        pos_targets=self.labels['gold_pos'],
                                                                        embeddings=self.embeddings,
                                                                        dep_global_step=self.dep_global_step)

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

            self.optimize_op_pre, self.optimize_op_srl, self.loss_pre, self.loss_srl, self.pre_predictions, \
            self.srl_predictions, self.srl_targets, self.srl_lr = srl_op.pred_srl_op(
                concat_outputs_pos=concat_outputs_pos,
                concat_outputs_dep=concat_outputs_dep,
                tokens_to_keep=self.tokens_to_keep,
                model_config=self.model_config,
                hparams=self.hparams,
                vocab=self.vocab,
                mode=self.mode,
                pos_probabilities=self.pos_probabilities,
                pos_targets=self.labels['gold_pos'],
                rel_targets=self.labels['parse_label'],
                head_targets=self.labels['parse_head'],
                rel_probabilities=self.rel_probabilities,
                head_probabilities=self.head_probabilites,
                t_pre=labels['predicate'],
                t_srl=labels['srl'],
                pos_num_heads=self.pos_num_heads,
                dep_num_heads=self.dep_num_heads,
                transition_params=transition_params,
                embeddings=self.embeddings,
                pre_global_step=self.pre_global_step,
                srl_global_step=self.srl_global_step)

        print('***Model built***')
