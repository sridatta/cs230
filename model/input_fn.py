import tensorflow as tf
import numpy as np

def load_tweets(fname, vocab):
    tweet_arr = np.load(fname)
    tweets = tf.data.Dataset.from_tensor_slices(tweet_arr)
    tweets = (tweets.map(lambda s: tf.string_split([s]).values)
                    .map(lambda t: (vocab.lookup(t), tf.size(t))))
    return tweets

def load_labels(fname):
    return tf.data.Dataset.from_tensor_slices(np.load(fname))

def input_fn(tweets, labels, buffer_size=1, batch_size=None):
    dataset = tf.data.Dataset.zip((tweets, labels))
    padded_shapes = (
        (tf.TensorShape([None]), # Pad the tweets
         tf.TensorShape([])), # Sentence length (scalar)
        tf.TensorShape([]) # Label (scalar)
    )
    dataset = (dataset
        .shuffle(buffer_size=buffer_size)
        .padded_batch(batch_size, padded_shapes)
        .prefetch(1))
    iterator = dataset.make_initializable_iterator()
    ((tweets, lengths), labels) = iterator.get_next()
    return iterator.initializer, tweets, lengths, labels