import tensorflow as tf

def build_model(mode, inputs, params, reuse=False):
    vocab = inputs["vocab"]
    lambd = inputs["l2_lambda"]
    keep_prob = inputs["keep_prob"]
    reg = tf.contrib.layers.l2_regularizer(lambd)
    embeddings = tf.constant(inputs["glove_weights"], tf.float32, name="embedding")
    if params.get("trainable_weights", False):
        embeddings += tf.get_variable(
            "embedding_delta", shape=[params["vocab_size"], 300], dtype=tf.float32, initializer=tf.zeros_initializer())
    sentence = tf.nn.embedding_lookup(embeddings, vocab.lookup(inputs["tweets"]), name="lookup")
    if params.get("cell_type") == "gru":
        cell = tf.contrib.rnn.GRUCell(params["lstm_num_units"], reuse=reuse)
        cell = tf.contrib.rnn.DropoutWrapper(cell, keep_prob)
        _, state  = tf.nn.dynamic_rnn(cell, sentence, sequence_length=inputs["lengths"], dtype=tf.float32)
    elif params.get("cell_type") == "bidi_gru":
        fw_cell = tf.contrib.rnn.GRUCell(params["lstm_num_units"], reuse=reuse)
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, keep_prob)
        bw_cell = tf.contrib.rnn.GRUCell(params["lstm_num_units"], reuse=reuse)
        bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, keep_prob)
        _, (fw_state, bw_state)  = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, sentence, sequence_length=inputs["lengths"], dtype=tf.float32)
        state = tf.concat([fw_state, bw_state], axis=1)
    elif params.get("cell_type") == "bidi_lstm":
        fw_cell = tf.contrib.rnn.LSTMCell(params["lstm_num_units"], reuse=reuse)
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, keep_prob)
        bw_cell = tf.contrib.rnn.LSTMCell(params["lstm_num_units"], reuse=reuse)
        bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, keep_prob)
        _, (fw_state, bw_state)  = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, sentence, sequence_length=inputs["lengths"], dtype=tf.float32)
        state = tf.concat([fw_state.c, bw_state.c], axis=1)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(params["lstm_num_units"], reuse=reuse)
        cell = tf.contrib.rnn.DropoutWrapper(cell, keep_prob)
        _, state_tuple  = tf.nn.dynamic_rnn(cell, sentence, sequence_length=inputs["lengths"], dtype=tf.float32)
        state = state_tuple.c

    logits = tf.layers.dense(state, 1, kernel_regularizer=reg) +  tf.constant(1e-8) # Output size 2 = binary classification
    return tf.squeeze(logits), (keep_prob, lambd)

def model_fn(mode, inputs, params, reuse=False):
    is_training = (mode == 'train')
    labels = inputs['labels']
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits, (keep_prob, lambd) = build_model(mode, inputs, params, reuse=reuse)
        tf.summary.histogram("logit_summary", logits)
        predictions = tf.nn.sigmoid(logits, name="predictions")

    binarized_labels = tf.equal(labels, 1.0)
    binarized_predictions = tf.greater(predictions, 0.5)
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(losses) + tf.losses.get_regularization_loss()
    accuracy = tf.reduce_mean(tf.cast(tf.equal(binarized_labels, binarized_predictions), tf.float32))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    if is_training:
        optimizer = tf.train.AdamOptimizer(params["learning_rate"])
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=binarized_labels, predictions=binarized_predictions),
            'loss': tf.metrics.mean(loss),
            'auc': tf.metrics.auc(labels=binarized_labels, predictions=predictions),
            'precision': tf.metrics.precision(labels=binarized_labels, predictions=binarized_predictions),
            'recall': tf.metrics.recall(labels=binarized_labels, predictions=binarized_predictions)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    model_spec = inputs.copy()
    model_spec['variable_init_op'] = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    model_spec['keep_prob'] = keep_prob
    model_spec['l2_lambda'] = lambd
    if is_training:
        model_spec['train_op'] = train_op
    return model_spec