import tensorflow as tf
import numpy as np
from generate_vocab import tokenize

def pad(arr, length, pad_val):
    if len(arr) >= length:
        return arr
    return arr + [pad_val]*(length - len(arr))

def load_tweets_naive(fname, max_len):
    tweet_arr = np.load(fname)
    len_fn = np.vectorize(lambda x: len(tokenize(x)))
    tweet_lens = len_fn(tweet_arr)

    tweet_arr = np.array([pad(tokenize(x), max_len, "<PAD>") for x in tweet_arr])
    return tweet_arr, tweet_lens

def load_labels_naive(fname):
    binarize = np.vectorize(lambda x: 1 if x == 1 else 0)
    return binarize(np.load(fname)).astype(np.float32)

def make_inputs(vocab, glove_weights, max_len):
    with tf.name_scope("input"):
        tweet_input = tf.placeholder(tf.string, [None, max_len], name="tweets")
        lens_input = tf.placeholder(tf.int64, [None], name="lengths")
        labels_input = tf.placeholder(tf.float32, [None], name="labels")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        lambd = tf.placeholder(tf.float32, name="l2_lambda")
    return {
        "tweets": tweet_input,
        "lengths": lens_input,
        "labels": labels_input,
        "vocab": vocab,
        'glove_weights': glove_weights,
        "l2_lambda": lambd,
        "keep_prob": keep_prob
    }