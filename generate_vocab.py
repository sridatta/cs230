import numpy as np
from collections import Counter
import string
import json

def add_to_vocab(tweets, vocab):
    for tweet in tweets:
        words = tweet.split(' ')
        vocab.update(w for w in words if w != ' ')

counts = Counter()
add_to_vocab(np.load("data/train/text.npy"), counts)
add_to_vocab(np.load("data/dev/text.npy"), counts)
add_to_vocab(np.load("data/test/text.npy"), counts)

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}
for word, _ in counts.most_common(50000):
   vocab[word] = len(vocab)

with open("data/vocab.json", 'w') as f:
    json.dump(vocab, f, indent=4, separators=(',', ': '))