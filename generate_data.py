import numpy as np
import pandas as pd
import os.path

def save_dataset(dataset, path):
    text = dataset["text"].values
    np.save(os.path.join(path, "text.npy"), text)
    labels = dataset["party"].values
    np.save(os.path.join(path, "labels.npy"), labels)

data = pd.read_pickle("data/tweets.pkl")
data = data[pd.notnull(data["text"])]

np.random.seed(230)
shuffle_idx = np.random.permutation(len(data))
data_shuffled = data.iloc[shuffle_idx]

train = data_shuffled[:int(0.7*len(data))]
dev = data_shuffled[int(0.7*len(data)):int(0.85*len(data))]
test = data_shuffled[int(0.85*len(data)):int(len(data))]

save_dataset(train, "data/train")
save_dataset(dev, "data/dev")
save_dataset(test, "data/test")