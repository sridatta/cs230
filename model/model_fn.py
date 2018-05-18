import tensorflow as tf

def build_model(mode, inputs, params):
    vocab = inputs["vocab"]
    embeddings = tf.constant(inputs["glove_weights"], tf.float32, name="embedding")
    sentence = tf.nn.embedding_lookup(embeddings, vocab.lookup(inputs["tweets"]), name="lookup")
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params["lstm_num_units"])
    _, state  = tf.nn.dynamic_rnn(lstm_cell, sentence, sequence_length=inputs["lengths"], dtype=tf.float32)
    logits = tf.layers.dense(state.h, 1) +  tf.constant(1e-8) # Output size 2 = binary classification
    return logits

def model_fn(mode, inputs, params, reuse=False):
    is_training = (mode == 'train')
    labels = inputs['labels']
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = tf.squeeze(build_model(mode, inputs, params))
        tf.summary.histogram("logit_summary", logits)
        predictions = tf.nn.sigmoid(logits, name="predictions")

    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(losses)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.equal(labels, 1.0), tf.greater(predictions, 0.5)), tf.float32))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    if is_training:
        optimizer = tf.train.AdamOptimizer(params["learning_rate"])
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=tf.equal(labels, 1.0), predictions=tf.greater(predictions, 0.5)),
            'loss': tf.metrics.mean(loss)
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
    if is_training:
        model_spec['train_op'] = train_op
    return model_spec