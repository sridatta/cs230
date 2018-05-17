import tensorflow as tf
import numpy as np
from generate_vocab import tokenize

def load_tweets(fname, vocab):
    tweet_arr = np.load(fname)
    clean_fn = np.vectorize(lambda x: ' '.join(tokenize(x)))
    tweet_arr = clean_fn(tweet_arr)
    tweets = tf.data.Dataset.from_tensor_slices(tweet_arr)
    tweets = (tweets
                    .map(lambda s: tf.string_split([s]).values)
                    .map(lambda t: (vocab.lookup(t), tf.size(t))))
    return tweets

def load_labels(fname):
    binarize = np.vectorize(lambda x: 1 if x == 1 else 0)
    data = binarize(np.load(fname)).astype(np.float32)
    return tf.data.Dataset.from_tensor_slices(data)

def input_fn(mode, tweets, labels, params):
    is_training = (mode == 'train')
    buffer_size = params["buffer_size"] if is_training else 1
    dataset = tf.data.Dataset.zip((tweets, labels))
    padded_shapes = (
        (tf.TensorShape([None]), # Pad the tweets
         tf.TensorShape([])), # Sentence length (scalar)
        tf.TensorShape([]) # Label (scalar)
    )
    dataset = (dataset
        .shuffle(buffer_size=buffer_size)
        .padded_batch(params["batch_size"], padded_shapes)
        .prefetch(1))
    iterator = dataset.make_initializable_iterator()
    ((tweets, lengths), labels) = iterator.get_next()
    return {
        'iterator_init_op': iterator.initializer,
        'tweets': tweets,
        'lengths': lengths,
        'labels': labels}