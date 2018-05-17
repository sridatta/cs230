import tensorflow as tf

def build_model(mode, inputs, params):
    embeddings = tf.constant(inputs["glove_weights"], tf.float32)
    sentence = tf.nn.embedding_lookup(embeddings, inputs["tweets"])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params["lstm_num_units"])
    _, state  = tf.nn.dynamic_rnn(lstm_cell, sentence, sequence_length=inputs["lengths"], dtype=tf.float32)
    logits = tf.layers.dense(state[0], 1) # Output size 2 = binary classification
    return logits

def model_fn(mode, inputs, params):
    is_training = (mode == 'train')
    labels = inputs['labels']
    with tf.variable_scope('model'):
        # Compute the output distribution of the model and the predictions
        logits = build_model(mode, inputs, params)
        predictions = tf.nn.sigmoid(logits)

    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(logits), labels=labels)
    loss = tf.reduce_mean(losses)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    if is_training:
        optimizer = tf.train.AdamOptimizer(params["learning_rate"])
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    model_spec = inputs.copy()
    model_spec['variable_init_op'] = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    if is_training:
        model_spec['train_op'] = train_op
    return model_spec