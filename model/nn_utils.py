import tensorflow as tf


def is_trainable(variable):
    return variable in tf.trainable_variables()


def leaky_relu(x):
    return tf.maximum(0.1 * x, x)


def set_vars_to_moving_average(moving_averager):
    moving_avg_variables = tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
    return tf.group(*[tf.assign(x, moving_averager.average(x)) for x in moving_avg_variables])


def layer_norm(inputs, epsilon=1e-6):
    """Applies layer normalization.

    Args:
        inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.

    Returns:
        A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.variable_scope("layer_norm"):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) * tf.rsqrt(variance + epsilon)
        outputs = gamma * normalized + beta
    return outputs


# def orthonormal_initializer(input_size, output_size):
#     """"""
#
#     if not tf.get_variable_scope().reuse:
#         print(tf.get_variable_scope().name)
#         I = np.eye(output_size)
#         lr = .1
#         eps = .05 / (output_size + input_size)
#         success = False
#         while not success:
#             Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
#             for i in range(100):
#                 QTQmI = Q.T.dot(Q) - I
#                 loss = np.sum(QTQmI ** 2 / 2)
#                 Q2 = Q ** 2
#                 Q -= lr * Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
#                 if np.isnan(Q[0, 0]):
#                     lr /= 2
#                     break
#             if np.isfinite(loss) and np.max(Q) < 1e6:
#                 success = True
#             eps *= 2
#         print('Orthogonal pretrainer loss: %.2e' % loss)
#     else:
#         print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
#         Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
#     return Q.astype(np.float32)


def linear_layer(inputs, output_size, add_bias=True, n_splits=1, initializer=None):
    """"""

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    output_size *= n_splits

    with tf.variable_scope('Linear'):
        # Reformat the input
        total_input_size = 0
        shapes = [a.get_shape().as_list() for a in inputs]
        for shape in shapes:
            total_input_size += shape[-1]
        input_shape = tf.shape(inputs[0])
        output_shape = []
        for i in range(len(shapes[0])):
            output_shape.append(input_shape[i])
        output_shape[-1] = output_size
        output_shape = tf.stack(output_shape)
        for i, (input_, shape) in enumerate(zip(inputs, shapes)):
            inputs[i] = tf.reshape(input_, [-1, shape[-1]])
        concatenation = tf.concat(axis=1, values=inputs)

        # Get the matrix
        if initializer is None:
            initializer = tf.initializers.orthogonal
            # mat = orthonormal_initializer(total_input_size, output_size // n_splits)
            # mat = np.concatenate([mat] * n_splits, axis=1)
            # initializer = tf.constant_initializer(mat)
        matrix = tf.get_variable('Weights', [total_input_size, output_size], initializer=initializer)
        # tf.add_to_collection('Weights', matrix)

        # Get the bias
        if add_bias:
            bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer())
        else:
            bias = 0

        # Do the multiplication
        new = tf.matmul(concatenation, matrix) + bias
        new = tf.reshape(new, output_shape)
        new.set_shape([tf.Dimension(None) for _ in range(len(shapes[0]) - 1)] + [tf.Dimension(output_size)])
        if n_splits > 1:
            return tf.split(axis=len(new.get_shape().as_list()) - 1, num_or_size_splits=n_splits, value=new)
        else:
            return new


# TODO clean this up
def MLP(inputs, output_size, func=leaky_relu, keep_prob=1.0, n_splits=1):
    """"""

    input_shape = inputs.get_shape().as_list()
    n_dims = len(input_shape)
    batch_size = tf.shape(inputs)[0]
    input_size = input_shape[-1]
    shape_to_set = [tf.Dimension(None)] * (n_dims - 1) + [tf.Dimension(output_size)]

    if keep_prob < 1:
        noise_shape = tf.stack([batch_size] + [1] * (n_dims - 2) + [input_size])
        inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)

    linear = linear_layer(inputs,
                          output_size,
                          n_splits=n_splits,
                          add_bias=True)
    if n_splits == 1:
        linear = [linear]
    for i, split in enumerate(linear):
        split = func(split)
        split.set_shape(shape_to_set)
        linear[i] = split
    if n_splits == 1:
        return linear[0]
    else:
        return linear


def bilinear(inputs1, inputs2, output_size, add_bias2=True, add_bias1=True, add_bias=False, initializer=None):
    """"""

    with tf.variable_scope('Bilinear'):
        # Reformat the inputs
        ndims = len(inputs1.get_shape().as_list())
        inputs1_shape = tf.shape(inputs1)
        inputs1_bucket_size = inputs1_shape[ndims - 2]
        inputs1_size = inputs1.get_shape().as_list()[-1]

        inputs2_shape = tf.shape(inputs2)
        inputs2_bucket_size = inputs2_shape[ndims - 2]
        inputs2_size = inputs2.get_shape().as_list()[-1]
        # output_shape = []
        batch_size1 = 1
        batch_size2 = 1
        for i in range(ndims - 2):
            batch_size1 *= inputs1_shape[i]
            batch_size2 *= inputs2_shape[i]
            # output_shape.append(inputs1_shape[i])
        # output_shape.append(inputs1_bucket_size)
        # output_shape.append(output_size)
        # output_shape.append(inputs2_bucket_size)
        # output_shape = tf.stack(output_shape)
        inputs1 = tf.reshape(inputs1, tf.stack([batch_size1, inputs1_bucket_size, inputs1_size]))
        inputs2 = tf.reshape(inputs2, tf.stack([batch_size2, inputs2_bucket_size, inputs2_size]))
        if add_bias1:
            inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size1, inputs1_bucket_size, 1]))])
        if add_bias2:
            inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size2, inputs2_bucket_size, 1]))])

        # Get the matrix
        if initializer is None:
            # mat = orthonormal_initializer(inputs1_size + add_bias1, inputs2_size + add_bias2)[:, None, :]
            # mat = np.concatenate([mat] * output_size, axis=1)
            # initializer = tf.constant_initializer(mat)
            initializer = tf.initializers.orthogonal
        weights = tf.get_variable('Weights', [inputs1_size + add_bias1, output_size, inputs2_size + add_bias2],
                                  initializer=initializer)
        # tf.add_to_collection('Weights', weights)

        # inputs1: num_triggers_in_batch x 1 x self.trigger_mlp_size
        # inputs2: batch x seq_len x self.role_mlp_size

        # Do the multiplications
        # (bn x d) (d x rd) -> (bn x rd)
        lin = tf.matmul(tf.reshape(inputs1, [-1, inputs1_size + add_bias1]),
                        tf.reshape(weights, [inputs1_size + add_bias1, -1]))
        # (b x nr x d) (b x n x d)T -> (b x nr x n)
        lin_reshape = tf.reshape(lin,
                                 tf.stack([batch_size1, inputs1_bucket_size * output_size, inputs2_size + add_bias2]))
        bilin = tf.matmul(lin_reshape, inputs2, adjoint_b=True)
        # (bn x r x n)
        bilin = tf.reshape(bilin, tf.stack([-1, output_size, inputs2_bucket_size]))

        # Get the bias
        if add_bias:
            bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer())
            bilin += tf.expand_dims(bias, 1)

        return bilin


def bilinear_classifier(inputs1, inputs2, keep_prob, add_bias1=True, add_bias2=False):
    """"""

    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]

    if keep_prob < 1:
        noise_shape = [batch_size, 1, input_size]
        inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape)
        inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape)

    bilin = bilinear(inputs1, inputs2, 1,
                     add_bias1=add_bias1,
                     add_bias2=add_bias2,
                     initializer=tf.zeros_initializer())
    output = tf.reshape(bilin, [batch_size, bucket_size, bucket_size])
    # output = tf.squeeze(bilin)
    return output


def bilinear_classifier_nary(inputs1, inputs2, n_classes, keep_prob, add_bias1=True, add_bias2=True):
    """"""

    input_shape1 = tf.shape(inputs1)
    input_shape2 = tf.shape(inputs2)

    batch_size1 = input_shape1[0]
    batch_size2 = input_shape2[0]

    # with tf.control_dependencies([tf.assert_equal(input_shape1[1], input_shape2[1])]):
    bucket_size1 = input_shape1[1]
    bucket_size2 = input_shape2[1]
    input_size1 = inputs1.get_shape().as_list()[-1]
    input_size2 = inputs2.get_shape().as_list()[-1]

    input_shape_to_set1 = [tf.Dimension(None), tf.Dimension(None), input_size1 + 1]
    input_shape_to_set2 = [tf.Dimension(None), tf.Dimension(None), input_size2 + 1]

    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
        noise_shape1 = tf.stack([batch_size1, 1, input_size1])
        noise_shape2 = tf.stack([batch_size2, 1, input_size2])

        inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape1)
        inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape2)

    inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size1, bucket_size1, 1]))])
    inputs1.set_shape(input_shape_to_set1)
    inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size2, bucket_size2, 1]))])
    inputs2.set_shape(input_shape_to_set2)

    bilin = bilinear(inputs1, inputs2,
                     n_classes,
                     add_bias1=add_bias1,
                     add_bias2=add_bias2,
                     initializer=tf.zeros_initializer())

    return bilin


def conditional_bilinear_classifier(inputs1, inputs2, n_classes, probs, keep_prob, add_bias1=True, add_bias2=True):
    """"""

    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    input_shape_to_set = [tf.Dimension(None), tf.Dimension(None), input_size + 1]
    # output_shape = tf.stack([batch_size, bucket_size, n_classes, bucket_size])
    if len(probs.get_shape().as_list()) == 2:
        probs = tf.to_float(tf.one_hot(tf.to_int64(probs), bucket_size, 1, 0))
    else:
        probs = tf.stop_gradient(probs)

    if keep_prob < 1:
        noise_shape = tf.stack([batch_size, 1, input_size])
        inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape)
        inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape)

    inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size, bucket_size, 1]))])
    inputs1.set_shape(input_shape_to_set)
    inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size, bucket_size, 1]))])
    inputs2.set_shape(input_shape_to_set)

    bilin = bilinear(inputs1, inputs2,
                     n_classes,
                     add_bias1=add_bias1,
                     add_bias2=add_bias2,
                     initializer=tf.zeros_initializer())
    bilin = tf.reshape(bilin, [batch_size, bucket_size, n_classes, bucket_size])
    weighted_bilin = tf.squeeze(tf.matmul(bilin, tf.expand_dims(probs, 3)), -1)

    return weighted_bilin, bilin


def parse_bilinear(hparams, inputs, targets, tokens_to_keep):
    with tf.variable_scope('parse_bilinear'):
        with tf.variable_scope('MLP'):
            dep_mlp, head_mlp = MLP(inputs, hparams.class_mlp_size + hparams.attn_mlp_size, n_splits=2,
                                    keep_prob=hparams.mlp_dropout)
            dep_arc_mlp, dep_rel_mlp = dep_mlp[:, :, :hparams.attn_mlp_size], dep_mlp[:, :, hparams.attn_mlp_size:]
            head_arc_mlp, head_rel_mlp = head_mlp[:, :, :hparams.attn_mlp_size], head_mlp[:, :, hparams.attn_mlp_size:]

        with tf.variable_scope('Arcs'):
            arc_logits = bilinear_classifier(dep_arc_mlp, head_arc_mlp, hparams.bilinear_dropout)

        num_tokens = tf.reduce_sum(tokens_to_keep)

        predictions = tf.argmax(arc_logits, -1)
        probabilities = tf.nn.softmax(arc_logits)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arc_logits, labels=targets)
        loss = tf.reduce_sum(cross_entropy * tokens_to_keep) / num_tokens

        # accuracy = tf.metrics.accuracy(targets, predictions, weights=tokens_to_keep)

        # outputs = {
        #     'loss': loss,
        #     'predictions': predictions,
        #     'probabilities': probabilities,
        #     'scores': arc_logits,
        #     'dep_rel_mlp': dep_arc_mlp,
        #     'head_rel_mlp': head_rel_mlp
        # }

    return loss, predictions, probabilities, arc_logits, dep_rel_mlp, head_rel_mlp  # , accuracy


def conditional_bilinear(mode, hparams, targets, num_labels, tokens_to_keep, dep_rel_mlp, head_rel_mlp,
                         parse_preds_train, parse_preds_eval):
    parse_preds = parse_preds_train if mode == 'TRAIN' else parse_preds_eval
    with tf.variable_scope('conditional_bilinear'):
        logits, _ = conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, num_labels,
                                                    parse_preds, hparams.bilinear_dropout)
    num_tokens = tf.reduce_sum(tokens_to_keep)

    predictions = tf.argmax(logits, -1)
    probabilities = tf.nn.softmax(logits)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    loss = tf.reduce_sum(cross_entropy * tokens_to_keep) / num_tokens

    # accuracy = tf.metrics.accuracy(targets, predictions, weights=tokens_to_keep)

    return loss, predictions, probabilities, logits #, accuracy


def softmax_classifier(hparams, inputs, targets, num_labels, tokens_to_keep):
    with tf.name_scope('softmax_classifier'):
        # todo add projection
        # projection_dim = model_config['predicate_pred_mlp_size']
        projection_dim = 200
        with tf.variable_scope('MLP'):
            mlp = MLP(inputs, projection_dim, keep_prob=hparams.mlp_dropout, n_splits=1)
        with tf.variable_scope('Classifier'):
            logits = MLP(mlp, num_labels, keep_prob=hparams.mlp_dropout, n_splits=1)

        targets_onehot = tf.one_hot(indices=targets, depth=num_labels, axis=-1)
        loss = tf.losses.softmax_cross_entropy(logits=tf.reshape(logits, [-1, num_labels]),
                                               onehot_labels=tf.reshape(targets_onehot, [-1, num_labels]),
                                               weights=tf.reshape(tokens_to_keep, [-1]),
                                               label_smoothing=hparams.label_smoothing,
                                               reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        probabilites = tf.nn.softmax(logits)
        # loss = loss / tf.reduce_sum(tokens_to_keep)
        #accuracy = tf.metrics.accuracy(targets, predictions, weights=tokens_to_keep)

    return loss, predictions, logits, probabilites #, accuracy


def srl_bilinear_classifier(mode, hparams, inputs, targets, num_labels, tokens_to_keep, predicate_preds_train,
                            predicate_preds_eval, predicate_targets, transition_params):
    with tf.name_scope('srl_bilinear'):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        batch_seq_len = input_shape[1]

        predicate_preds = predicate_preds_train if mode == 'TRAIN' else predicate_preds_eval

        with tf.variable_scope('MLP'):
            predicate_role_mlp = MLP(inputs, hparams.predicate_mlp_size + hparams.role_mlp_size,
                                     keep_prob=hparams.mlp_dropout)
            predicate_mlp, role_mlp = predicate_role_mlp[:, :, :hparams.predicate_mlp_size], \
                                      predicate_role_mlp[:, :, hparams.predicate_mlp_size:]

        with tf.variable_scope('Bilinear'):
            predicate_gather_indices = tf.where(tf.equal(predicate_preds, 1))
            gathered_predicates = tf.expand_dims(tf.gather_nd(predicate_mlp, predicate_gather_indices), 1)
            tiled_roles = tf.reshape(tf.tile(role_mlp, [1, batch_seq_len, 1]),
                                     [batch_size, batch_seq_len, batch_seq_len, hparams.role_mlp_size])
            gathered_roles = tf.gather_nd(tiled_roles, predicate_gather_indices)
            srl_logits = bilinear_classifier_nary(gathered_predicates, gathered_roles, num_labels,
                                                  hparams.bilinear_dropout)
            srl_logits_transposed = tf.transpose(srl_logits, [0, 2, 1])

        mask_tiled = tf.reshape(tf.tile(tokens_to_keep, [1, batch_seq_len]), [batch_size, batch_seq_len, batch_seq_len])
        mask = tf.gather_nd(mask_tiled, tf.where(tf.equal(predicate_preds, 1)))

        srl_targets_transposed = tf.transpose(targets, [0, 2, 1])

        gold_predicate_counts = tf.reduce_sum(predicate_targets, -1)
        srl_targets_indices = tf.where(tf.sequence_mask(tf.reshape(gold_predicate_counts, [-1])))

        srl_targets_gold_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_indices)

        predicted_predicate_counts = tf.reduce_sum(predicate_preds, -1)
        srl_targets_pred_indices = tf.where(tf.sequence_mask(tf.reshape(predicted_predicate_counts, [-1])))
        srl_targets_predicted_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_pred_indices)

        predictions = tf.cast(tf.argmax(srl_logits_transposed, axis=-1), tf.int32)

        seq_lens = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

        if transition_params is not None and (mode == 'PREDICT' or mode == 'EVAL'):
            predictions, score = tf.contrib.crf.crf_decode(srl_logits_transposed, transition_params, seq_lens)

        if transition_params is not None and mode == 'TRAIN' and transition_params in tf.trainable_variables():
            # flat_seq_lens = tf.reshape(tf.tile(seq_lens, [1, bucket_size]), tf.stack([batch_size * bucket_size]))
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(srl_logits_transposed,
                                                                                  srl_targets_predicted_predicates,
                                                                                  seq_lens, transition_params)
            loss = tf.reduce_mean(-log_likelihood)
        else:
            srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
            loss = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                   onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                   weights=tf.reshape(mask, [-1]),
                                                   label_smoothing=hparams.label_smoothing,
                                                   reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        #corrects = tf.reduce_sum(tf.cast(tf.equal(predictions, srl_targets_gold_predicates), tf.float32))

        return loss, predictions, srl_logits_transposed, srl_targets_gold_predicates  #, corrects


def label_embedding_fn(label_scores, label_embeddings):
    with tf.variable_scope('label_embedding_fn'):
        embeddings_shape = label_embeddings.get_shape()
        vocab_size = embeddings_shape[0]
        label_embedding_dim = embeddings_shape[1]
        input_shape = tf.shape(label_scores)
        batch_size = input_shape[0]
        batch_seq_len = input_shape[1]

        # check whether this thing is actually scores or if it's predictions, and needs
        # to be expanded out to one-hot scores. If it's actually scores, dims should be
        # batch x batch_seq_len x num_classes, and thus rank should be 3
        if len(label_scores.get_shape()) < 3:
            label_scores = tf.one_hot(label_scores, vocab_size)

        label_scores = tf.reshape(label_scores, [-1, vocab_size])
        label_embeddings = tf.reshape(label_embeddings, [vocab_size, label_embedding_dim])
        averaged = tf.matmul(label_scores, label_embeddings)

        return tf.reshape(averaged, [batch_size, batch_seq_len, label_embedding_dim])


def get_separate_scores_preds_from_joint(joint_outputs, joint_maps, joint_num_labels):
    predictions = joint_outputs['predictions']
    scores = joint_outputs['scores']
    output_shape = tf.shape(predictions)
    batch_size = output_shape[0]
    batch_seq_len = output_shape[1]
    sep_outputs = {}
    for map_name, label_comp_map in joint_maps.items():
        short_map_name = map_name.split('_to_')[-1]
        label_comp_predictions = tf.nn.embedding_lookup(label_comp_map, predictions)
        sep_outputs["%s_predictions" % short_map_name] = tf.squeeze(label_comp_predictions, -1)

        # marginalize out probabilities for this task
        task_num_labels = tf.shape(tf.unique(tf.reshape(label_comp_map, [-1]))[0])[0]
        joint_probabilities = tf.nn.softmax(scores)
        joint_probabilities_flat = tf.reshape(joint_probabilities, [-1, joint_num_labels])
        segment_ids = tf.squeeze(tf.nn.embedding_lookup(label_comp_map, tf.range(joint_num_labels)), -1)
        segment_scores = tf.unsorted_segment_sum(tf.transpose(joint_probabilities_flat), segment_ids, task_num_labels)
        segment_scores = tf.reshape(tf.transpose(segment_scores), [batch_size, batch_seq_len, task_num_labels])
        sep_outputs["%s_probabilities" % short_map_name] = segment_scores
    return sep_outputs


def joint_softmax_classifier(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, joint_maps,
                             transition_params):
    with tf.name_scope('joint_softmax_classifier'):
        # todo pass this as initial proj dim (which is optional)
        projection_dim = model_config['predicate_pred_mlp_size']

        with tf.variable_scope('MLP'):
            mlp = MLP(inputs, projection_dim, keep_prob=hparams.mlp_dropout, n_splits=1)
        with tf.variable_scope('Classifier'):
            logits = MLP(mlp, num_labels, keep_prob=hparams.mlp_dropout, n_splits=1)

        # todo implement this
        if transition_params is not None:
            print('Transition params not yet supported in joint_softmax_classifier')
            exit(1)

        # logits = tf.clip_by_value(logits, 1e-8, 1.0)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

        cross_entropy *= tokens_to_keep
        loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(tokens_to_keep)

        predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        output = {
            'loss': loss,
            'predictions': predictions,
            'scores': logits
        }

        # now get separate-task scores and predictions for each of the maps we've passed through
        separate_output = get_separate_scores_preds_from_joint(output, joint_maps, num_labels)
        combined_output = {**output, **separate_output}

        return combined_output
