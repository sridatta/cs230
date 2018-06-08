import numpy as np
from collections import Counter
import string
import json

def replace_word(w):
    if w.startswith("http"):
        return "<LINK>"
    if w.startswith("#"):
        return w[1:]
    if w.startswith("@"):
        return w
    if w.startswith(".@"):
        return w[1:]
    return w.strip().strip(string.punctuation).lower()


def tokenize(text):
    words = text.split()
    return [replace_word(w) for w in words]

def vocab_as_sorted_list(vocab):
    return [p[0]for p in sorted(vocab.items(), key=lambda p: p[1])]

def add_to_vocab(tweets, counts, skip_words):
    for tweet in tweets:
        for w in tokenize(tweet):
            counts.update([w])

def generate_glove_weights(vocab):
    # Get vocab sorted by frequency
    vocab_list = vocab_as_sorted_list(vocab)
    embeddings_index = {}
    with open("data/glove.6B/glove.6B.300d.txt") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(vocab_list), 300))
    hits = 0.0
    for i, word in enumerate(vocab_list):
        if word.lower() in embeddings_index:
            hits += 1
            embedding_matrix[i] = embeddings_index[word.lower()]
        else:
            embedding_matrix[i] = np.random.randn(1, 300)
    print("Word hit rate %f" % (hits/len(vocab_list)))
    return embedding_matrix

if __name__ == "__main__":
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<LINK>": 2,
        "<HASHTAG>": 3,
        "<MENTION>": 4
    }
    counts = Counter()
    train = np.load("data/train/text.npy")
    add_to_vocab(train, counts, set(vocab.keys()))

    dev = np.load("data/dev/text.npy")
    add_to_vocab(dev, counts, set(vocab.keys()))

    test = np.load("data/test/text.npy")
    add_to_vocab(test, counts, set(vocab.keys()))

    for word, count in counts.most_common():
        if count == 1:
            break
        vocab[word] = len(vocab)

    with open("data/vocab.json", 'w') as f:
        json.dump(vocab, f, indent=4, separators=(',', ': '))

    glove = generate_glove_weights(vocab)
    np.save("data/glove.npy", glove)

    params = {}
    params["vocab_pad_word"] = "<PAD>"
    params["vocab_pad_idx"] = 0
    params["vocab_unk_word"] = "<UNK>"
    params["vocab_unk_idx"] = 1
    params["vocab_size"] = len(vocab)
    params["train_set_size"] = len(train)
    params["dev_set_size"] = len(dev)
    params["test_set_size"] = len(test)
    with open('data/dataset_params.json', "w") as f:
        json.dump(params, f, indent=4, separators=(',', ': '))