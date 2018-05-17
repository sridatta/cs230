import tensorflow as tf
import numpy as np
from model.input_fn import input_fn
from model.input_fn import load_labels
from model.input_fn import load_tweets
from generate_vocab import vocab_as_sorted_list
import json
import params as params_util
from model.model_fn import model_fn

if __name__ == "__main__":
    params = params_util.load_params()
    print(params)
    tf.set_random_seed(230)
    with open("data/vocab.json") as f:
        vocab_json = json.load(f)
    vocab_list = vocab_as_sorted_list(vocab_json)
    vocab = tf.contrib.lookup.index_table_from_tensor(tf.constant(vocab_list), default_value=params["vocab_unk_idx"])

    train_tweets = load_tweets("data/train/text.npy", vocab)
    train_labels = load_labels("data/train/labels.npy")

    dev_tweets = load_tweets("data/dev/text.npy", vocab)
    dev_labels = load_labels("data/dev/labels.npy")

    glove_weights = np.load("data/glove.npy")
    params["buffer_size"] = params["train_set_size"]
    train_inputs = input_fn("train", train_tweets, train_labels, params)
    train_inputs["glove_weights"] = glove_weights

    dev_inputs = input_fn("eval", dev_tweets, dev_labels, params)
    dev_inputs["glove_weights"] = glove_weights

    train_model = model_fn("train", train_inputs, params)
    # eval_model = model_fn("eval", dev_inputs, params)
