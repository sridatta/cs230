import tensorflow as tf
import numpy as np
from model.input_fn import input_fn
from model.input_fn import load_labels
from model.input_fn import load_tweets
import json

def load_glove_weights(vocab_list):
    embeddings_index = {}
    with open("data/glove.6B/glove.6B.100d.txt") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(vocab_list), 100))
    for i, word in enumerate(vocab_list):
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
    return tf.constant(embedding_matrix)

if __name__ == "__main__":
    tf.set_random_seed(230)
    with open("data/vocab.json") as f:
        vocab_list = json.load(f)
    vocab_list = [p[0]for p in sorted(vocab_list.items(), key=lambda p: p[1])]
    vocab = tf.contrib.lookup.index_table_from_tensor(tf.constant(vocab_list), default_value=1)

    train_tweets = load_tweets("data/train/text.npy", vocab)
    train_labels = load_labels("data/train/labels.npy")

    dev_tweets = load_tweets("data/dev/text.npy", vocab)
    dev_labels = load_labels("data/dev/labels.npy")

    glove_weights = load_glove_weights(vocab_list)

    (iter_init, tweets, lengths, labels) = input_fn(train_tweets, train_labels, batch_size=10)