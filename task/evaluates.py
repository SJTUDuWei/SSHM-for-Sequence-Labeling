# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2019/1/31 下午9:21
# @Project: unified_chinese_multi_tasking_framework
# @File: evaluates.py
# @Software: PyCharm


import tensorflow as tf
import numpy as np
import time
import os
from subprocess import check_output, CalledProcessError
import re


def convert_bilou(bio_predicted_roles):
    """

  :param bio_predicted_roles: sequence of BIO-encoded predicted role labels
  :return: sequence of conll-formatted predicted role labels
  """

    converted = []
    started_types = []
    for i, s in enumerate(bio_predicted_roles):
        s = s.decode('utf-8')
        label_parts = s.split('/')
        curr_len = len(label_parts)
        combined_str = ''
        Itypes = []
        Btypes = []
        for idx, label in enumerate(label_parts):
            bilou = label[0]
            label_type = label[2:]
            props_str = ''
            if bilou == 'I':
                Itypes.append(label_type)
                props_str = ''
            elif bilou == 'O':
                curr_len = 0
                props_str = ''
            elif bilou == 'U':
                # need to check whether last one was ended
                props_str = '(' + label_type + ('*)' if idx == len(label_parts) - 1 else "")
            elif bilou == 'B':
                # need to check whether last one was ended
                props_str = '(' + label_type
                started_types.append(label_type)
                Btypes.append(label_type)
            elif bilou == 'L':
                props_str = ')'
                started_types.pop()
                curr_len -= 1
            combined_str += props_str
        while len(started_types) > curr_len:
            converted[-1] += ')'
            started_types.pop()
        while len(started_types) < len(Itypes) + len(Btypes):
            combined_str = '(' + Itypes[-1] + combined_str
            started_types.append(Itypes[-1])
            Itypes.pop()
        if not combined_str:
            combined_str = '*'
        elif combined_str[0] == "(" and combined_str[-1] != ")":
            combined_str += '*'
        elif combined_str[-1] == ")" and combined_str[0] != "(":
            combined_str = '*' + combined_str
        converted.append(combined_str)
    while len(started_types) > 0:
        converted[-1] += ')'
        started_types.pop()
    return converted


def accuracy(predictions, targets, mask):
    with tf.name_scope('accuracy'):
        return tf.metrics.accuracy(targets, predictions, weights=mask)


def batch_str_decode(string_array, codec='utf-8'):
    string_array = np.array(string_array)
    return np.reshape(np.array(list(map(lambda p: p if not p or isinstance(p, str) else p.decode(codec),
                                        np.reshape(string_array, [-1])))), string_array.shape)


def write_srl_eval_09(filename, words, predicates, sent_lens, role_labels, parse_heads, parse_labels, pos_tags):
    with open(filename, 'w') as f:
        role_labels_start_idx = 0

        predicates = batch_str_decode(predicates)
        words = batch_str_decode(words)
        parse_labels = batch_str_decode(parse_labels)
        pos_tags = batch_str_decode(pos_tags)
        role_labels = batch_str_decode(role_labels)

        # todo pretty sure this assumes that 0 == '_'
        num_predicates_per_sent = np.sum(predicates == 'True', -1)

        # for each sentence in the batch
        for sent_words, sent_predicates, sent_len, sent_num_predicates, \
            sent_parse_heads, sent_parse_labels, sent_pos_tags in zip(words, predicates, sent_lens,
                                                                      num_predicates_per_sent,
                                                                      parse_heads, parse_labels, pos_tags):
            # grab predicates and convert to conll format from bio
            # this is a sent_num_predicates x batch_seq_len array
            sent_role_labels = np.transpose(
                role_labels[role_labels_start_idx: role_labels_start_idx + sent_num_predicates])

            # this is a list of sent_num_predicates lists of srl role labels
            role_labels_start_idx += sent_num_predicates

            # for each token in the sentence
            for j, (word, predicate, parse_head, parse_label, pos_tag) in enumerate(zip(sent_words[:sent_len],
                                                                                        sent_predicates[:sent_len],
                                                                                        sent_parse_heads[:sent_len],
                                                                                        sent_parse_labels[:sent_len],
                                                                                        sent_pos_tags[:sent_len])):
                tok_role_labels = sent_role_labels[j] if len(sent_role_labels) > 0 else []
                predicate_str = "Y\t%s.%s" % (word, predicate) if predicate == 'True' else '_\t_'
                roles_str = '\t'.join(tok_role_labels)
                outputs = (
                    j, word, pos_tag, pos_tag, parse_head, parse_head, parse_label, parse_label, predicate_str,
                    roles_str)
                print("%s\t%s\t_\t_\t%s\t%s\t_\t_\t%s\t%s\t%s\t%s\t%s\t%s" % outputs, file=f)
            print(file=f)


def write_parse_eval(filename, words, parse_heads, sent_lens, parse_labels, pos_tags):
    with open(filename, 'w') as f:

        # for each sentence in the batch
        for sent_words, sent_parse_heads, sent_len, sent_parse_labels, sent_pos_tags in zip(words, parse_heads,
                                                                                            sent_lens,
                                                                                            parse_labels, pos_tags):
            # for each token in the sentence
            for j, (word, parse_head, parse_label, pos_tag) in enumerate(zip(sent_words[:sent_len],
                                                                             sent_parse_heads[:sent_len],
                                                                             sent_parse_labels[:sent_len],
                                                                             sent_pos_tags[:sent_len])):
                parse_head = 0 if j == parse_head else parse_head + 1
                token_outputs = (
                    j, word.decode('utf-8'), pos_tag.decode('utf-8'), int(parse_head), parse_label.decode('utf-8'))
                print("%d\t%s\t_\t%s\t_\t_\t%d\t%s" % token_outputs, file=f)
            print(file=f)


def write_srl_debug(filename, words, predicates, sent_lens, role_labels, pos_predictions, pos_targets):
    with open(filename, 'w') as f:
        role_labels_start_idx = 0
        num_predicates_per_sent = np.sum(predicates, -1)
        # for each sentence in the batch
        for sent_words, sent_predicates, sent_len, sent_num_predicates, pos_preds, pos_targs in zip(words, predicates,
                                                                                                    sent_lens,
                                                                                                    num_predicates_per_sent,
                                                                                                    pos_predictions,
                                                                                                    pos_targets):
            # grab predicates and convert to conll format from bio
            # this is a sent_num_predicates x batch_seq_len array
            sent_role_labels_bio = role_labels[role_labels_start_idx: role_labels_start_idx + sent_num_predicates]

            # this is a list of sent_num_predicates lists of srl role labels
            sent_role_labels = list(map(list, zip(*[convert_bilou(j[:sent_len]) for j in sent_role_labels_bio])))
            role_labels_start_idx += sent_num_predicates

            sent_role_labels_bio = list(zip(*sent_role_labels_bio))

            pos_preds = list(map(lambda d: d.decode('utf-8'), pos_preds))
            pos_targs = list(map(lambda d: d.decode('utf-8'), pos_targs))

            # for each token in the sentence
            # printed = False
            for j, (word, predicate, pos_p, pos_t) in enumerate(zip(sent_words[:sent_len], sent_predicates[:sent_len],
                                                                    pos_preds[:sent_len], pos_targs[:sent_len])):
                tok_role_labels = sent_role_labels[j] if sent_role_labels else []
                bio_tok_role_labels = sent_role_labels_bio[j][:sent_len] if sent_role_labels else []
                word_str = word.decode('utf-8')
                predicate_str = str(predicate)
                roles_str = '\t'.join(tok_role_labels)
                bio_roles_str = '\t'.join(map(lambda d: d.decode('utf-8'), bio_tok_role_labels))
                print("%s\t%s\t%s\t%s\t%s\t%s" % (word_str, predicate_str, pos_t, pos_p, roles_str, bio_roles_str),
                      file=f)
            print(file=f)


def conll09_srl_eval(srl_predictions, predicate_predictions, words, mask, srl_targets, predicate_targets,
                     parse_label_predictions, parse_head_predictions, parse_label_targets, parse_head_targets,
                     pos_targets, pos_predictions, pred_srl_eval_file, gold_srl_eval_file):

    # need to print for every word in every sentence
    sent_lens = np.sum(mask, -1).astype(np.int32)

    # write gold labels
    write_srl_eval_09(gold_srl_eval_file, words, predicate_targets, sent_lens, srl_targets, parse_head_targets,
                      parse_label_targets, pos_targets)

    # write predicted labels
    write_srl_eval_09(pred_srl_eval_file, words, predicate_predictions, sent_lens, srl_predictions,
                      parse_head_predictions, parse_label_predictions, pos_predictions)

    # run eval script
    labeled_correct, labeled_excess, labeled_missed, prop_correct, prop_excess, prop_missed = 0, 0, 0, 0, 0, 0
    with open(os.devnull, 'w') as devnull:
        try:
            srl_eval = check_output(["perl", "bin/eval09.pl", "-g", gold_srl_eval_file, "-s", pred_srl_eval_file],
                                    stderr=devnull)
            srl_eval = srl_eval.decode('utf-8')
            eval_lines = srl_eval.split('\n')
            labeled_precision_ints = list(map(int, re.sub('[^0-9 ]', '', eval_lines[7]).split()))
            labeled_recall_ints = list(map(int, re.sub('[^0-9 ]', '', eval_lines[8]).split()))
            prop_precision_ints = list(map(int, re.sub('[^0-9 ]', '', eval_lines[13]).split()))
            prop_recall_ints = list(map(int, re.sub('[^0-9 ]', '', eval_lines[14]).split()))

            labeled_correct = labeled_precision_ints[0] + labeled_precision_ints[1]
            labeled_excess = labeled_precision_ints[2] + labeled_precision_ints[3] - labeled_correct
            labeled_missed = labeled_recall_ints[2] + labeled_recall_ints[3] - labeled_correct

            prop_correct = prop_precision_ints[0]
            prop_excess = prop_precision_ints[1] - prop_correct
            prop_missed = prop_recall_ints[1] - prop_correct

        except CalledProcessError as e:
            tf.logging.log(tf.logging.ERROR, "Call to eval09.pl (conll09 srl eval) failed.")

    return labeled_correct, labeled_excess, labeled_missed


def create_metric_variable(name, shape, dtype):
    return tf.get_variable(name=name, shape=shape, dtype=dtype, trainable=False,
                           collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES])


def conll_srl_eval(predictions, targets, predicate_predictions, words, mask, predicate_targets, reverse_maps,
                   gold_srl_eval_file, pred_srl_eval_file, pos_predictions, pos_targets, parse_head_targets,
                   parse_head_predictions, parse_label_targets, parse_label_predictions):
    with tf.name_scope('conll_srl_eval'):
        # create accumulator variables
        # correct_count = create_metric_variable("correct_count", shape=[], dtype=tf.int64)
        # excess_count = create_metric_variable("excess_count", shape=[], dtype=tf.int64)
        # missed_count = create_metric_variable("missed_count", shape=[], dtype=tf.int64)
        # first, use reverse maps to convert ints to strings
        # todo order of map.values() is probably not guaranteed; should prob sort by keys first
        str_predictions = tf.nn.embedding_lookup(np.array(list(reverse_maps['srl'].values())), predictions)
        str_words = tf.nn.embedding_lookup(np.array(list(reverse_maps['word'].values())), words)
        str_targets = tf.nn.embedding_lookup(np.array(list(reverse_maps['srl'].values())), targets)

        str_parse_label_targets = tf.nn.embedding_lookup(np.array(list(reverse_maps['parse_label'].values())),
                                                         parse_label_targets)
        str_parse_label_predictions = tf.nn.embedding_lookup(np.array(list(reverse_maps['parse_label'].values())),
                                                             parse_label_predictions)

        str_pos_predictions = tf.nn.embedding_lookup(np.array(list(reverse_maps['gold_pos'].values())), pos_predictions)
        str_pos_targets = tf.nn.embedding_lookup(np.array(list(reverse_maps['gold_pos'].values())), pos_targets)

        str_predicate_predictions = tf.nn.embedding_lookup(np.array(list(reverse_maps['predicate'].values())),
                                                           predicate_predictions)
        str_predicate_targets = tf.nn.embedding_lookup(np.array(list(reverse_maps['predicate'].values())),
                                                       predicate_targets)

        # need to pass through the stuff for pyfunc
        # pyfunc is necessary here since we need to write to disk
        py_eval_inputs = [str_predictions, str_predicate_predictions, str_words, mask, str_targets,
                          str_predicate_targets,
                          str_parse_label_predictions, parse_head_predictions, str_parse_label_targets,
                          parse_head_targets,
                          str_pos_targets, str_pos_predictions, pred_srl_eval_file, gold_srl_eval_file]
        out_types = [tf.int64, tf.int64, tf.int64]
        correct, excess, missed = tf.py_func(conll09_srl_eval, py_eval_inputs, out_types, stateful=False)

        # update_correct_op = tf.assign_add(correct_count, correct)
        # update_excess_op = tf.assign_add(excess_count, excess)
        # update_missed_op = tf.assign_add(missed_count, missed)

        # precision_update_op = update_correct_op / (update_correct_op + update_excess_op)
        # recall_update_op = update_correct_op / (update_correct_op + update_missed_op)
        # f1_update_op = 2 * precision_update_op * recall_update_op / (precision_update_op + recall_update_op)

        # precision = float(correct) / (correct + excess)
        # recall = float(correct) / (correct + missed)
        # f1 = 2 * precision * recall / (precision + recall)

        # return f1 #, f1_update_op
        return correct, excess, missed


def conll_parse_eval_py(parse_label_predictions, parse_head_predictions, words, mask, parse_label_targets,
                        parse_head_targets, pred_eval_file, gold_eval_file, pos_targets):
    # need to print for every word in every sentence
    sent_lens = np.sum(mask, -1).astype(np.int32)

    # write gold labels
    write_parse_eval(gold_eval_file, words, parse_head_targets, sent_lens, parse_label_targets, pos_targets)

    # write predicted labels
    write_parse_eval(pred_eval_file, words, parse_head_predictions, sent_lens, parse_label_predictions, pos_targets)

    # run eval script
    total, labeled_correct, unlabeled_correct, label_correct = 0, 0, 0, 0
    with open(os.devnull, 'w') as devnull:
        try:
            eval = check_output(["perl", "bin/eval.pl", "-g", gold_eval_file, "-s", pred_eval_file], stderr=devnull)
            eval_str = eval.decode('utf-8')

            # Labeled attachment score: 26444 / 29058 * 100 = 91.00 %
            # Unlabeled attachment score: 27251 / 29058 * 100 = 93.78 %
            # Label accuracy score: 27395 / 29058 * 100 = 94.28 %
            first_three_lines = eval_str.split('\n')[:3]
            total = int(first_three_lines[0].split()[5])
            labeled_correct, unlabeled_correct, label_correct = map(lambda l: int(l.split()[3]), first_three_lines)
        except CalledProcessError as e:
            tf.logging.log(tf.logging.ERROR, "Call to eval.pl (conll parse eval) failed.")

    return total, np.array([labeled_correct, unlabeled_correct, label_correct], dtype=int)


def conll_parse_eval(predictions, targets, parse_head_predictions, words, mask, parse_head_targets, reverse_maps,
                     gold_parse_eval_file, pred_parse_eval_file, pos_targets):
    with tf.name_scope('conll_parse_eval'):
        # create accumulator variables
        # total_count = create_metric_variable("total_count", shape=[], dtype=tf.int64)
        # correct_count = create_metric_variable("correct_count", shape=[3], dtype=tf.int64)

        # first, use reverse maps to convert ints to strings
        # todo order of map.values() is probably not guaranteed; should prob sort by keys first
        str_words = tf.nn.embedding_lookup(np.array(list(reverse_maps['word'].values())), words)
        str_predictions = tf.nn.embedding_lookup(np.array(list(reverse_maps['parse_label'].values())), predictions)
        str_targets = tf.nn.embedding_lookup(np.array(list(reverse_maps['parse_label'].values())), targets)
        str_pos_targets = tf.nn.embedding_lookup(np.array(list(reverse_maps['gold_pos'].values())), pos_targets)

        # need to pass through the stuff for pyfunc
        # pyfunc is necessary here since we need to write to disk
        py_eval_inputs = [str_predictions, parse_head_predictions, str_words, mask, str_targets, parse_head_targets,
                          pred_parse_eval_file, gold_parse_eval_file, str_pos_targets]
        out_types = [tf.int64, tf.int64]
        total, corrects = tf.py_func(conll_parse_eval_py, py_eval_inputs, out_types, stateful=False)

        # update_total_count_op = tf.assign_add(total_count, total)
        # update_correct_op = tf.assign_add(correct_count, corrects)

        # update_op = update_correct_op / update_total_count_op

        # accuracies = corrects / total

        # return accuracies #, update_op
        return total, corrects


def conll09_srl_eval_np(predictions, targets, predicate_predictions, words, mask, predicate_targets, reverse_maps,
                        gold_srl_eval_file, pred_srl_eval_file, pos_predictions, pos_targets, parse_head_predictions,
                        parse_head_targets, parse_label_predictions, parse_label_targets, accumulator):
    # first, use reverse maps to convert ints to strings
    str_srl_predictions = [list(map(reverse_maps['srl'].get, s)) for s in predictions]
    str_words = [list(map(reverse_maps['word'].get, s)) for s in words]
    str_srl_targets = [list(map(reverse_maps['srl'].get, s)) for s in targets]
    str_pos_targets = [list(map(reverse_maps['gold_pos'].get, s)) for s in pos_targets]
    str_pos_predictions = [list(map(reverse_maps['gold_pos'].get, s)) for s in pos_predictions]
    str_parse_label_targets = [list(map(reverse_maps['parse_label'].get, s)) for s in parse_label_targets]
    str_parse_label_predictions = [list(map(reverse_maps['parse_label'].get, s)) for s in parse_label_predictions]
    str_predicate_predictions = [list(map(reverse_maps['predicate'].get, s)) for s in predicate_predictions]
    str_predicate_targets = [list(map(reverse_maps['predicate'].get, s)) for s in predicate_targets]

    correct, excess, missed = conll09_srl_eval(str_srl_predictions, str_predicate_predictions, str_words, mask,
                                               str_srl_targets, str_predicate_targets, str_parse_label_predictions,
                                               parse_head_predictions, str_parse_label_targets, parse_head_targets,
                                               str_pos_targets, str_pos_predictions, pred_srl_eval_file,
                                               gold_srl_eval_file)

    accumulator['correct'] += correct
    accumulator['excess'] += excess
    accumulator['missed'] += missed

    precision = accumulator['correct'] / (accumulator['correct'] + accumulator['excess'])
    recall = accumulator['correct'] / (accumulator['correct'] + accumulator['missed'])
    f1 = 2 * precision * recall / (precision + recall)

    return f1
