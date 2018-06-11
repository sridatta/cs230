import tensorflow as tf
import numpy as np
from model.input_fn import load_tweets_naive, load_labels_naive, make_inputs
from generate_vocab import vocab_as_sorted_list
import json
import params as params_util
from model.model_fn import model_fn
from tqdm import trange
import datetime
import math
import os.path
import os
import pandas as pd
from sklearn.metrics import confusion_matrix

# Some stupid thing to write np floats to json
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def shuffle_datasets(datasets):
    shuffle_idx = np.random.permutation(len(datasets[0]))
    return tuple(dataset[shuffle_idx] for dataset in datasets)

def evaluate_model(inputs, params, eval_model, dataset_size):
    batch_size = params["batch_size"]
    with tf.Session() as sess:
        sess.run(eval_model['variable_init_op'])
        saver = tf.train.Saver()
        save_dir = os.path.join(os.getcwd(), "results", params["run_id"])
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))
        num_minibatches =  math.ceil(dataset_size / float(batch_size))
        # Evaluate on dev set
        print("Evaluating train set...")
        sess.run(eval_model['metrics_init_op'])
        tweets_shuffle, lens_shuffle, labels_shuffle = shuffle_datasets([tweets, lens, labels])
        preds_accumulated = []
        for batch_num in range(num_minibatches):
            start_idx = batch_num*batch_size
            end_idx = min((batch_num+1)*batch_size, dataset_size)
            feed_dict = {
                inputs["tweets"]: tweets_shuffle[start_idx:end_idx],
                inputs["lengths"]: lens_shuffle[start_idx:end_idx],
                inputs["labels"]: labels_shuffle[start_idx:end_idx],
                inputs["keep_prob"]: 1.0,
                inputs["l2_lambda"]: params["l2_lambda"]
            }
            preds, _ = sess.run(
                [eval_model["predictions"],
                 eval_model["update_metrics"]], feed_dict)
            preds_accumulated.append(preds)
        dev_metric_values = sess.run({k: v[0] for k, v in eval_model["metrics"].items()})
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in dev_metric_values.items())
        print(metrics_string)
        preds_concat = np.concatenate(preds_accumulated, axis=-1)
        tweets_concat = np.apply_along_axis(lambda x: (' '.join(x)), axis=1, arr=tweets_shuffle)
        print(tweets_concat.shape, labels_shuffle.shape, preds_concat.shape)
        print(tweets_concat[0])
        d = {"tweets": tweets_concat.flatten(), "labels": labels_shuffle.flatten(), "predictions": preds_concat.flatten()}
        df = pd.DataFrame(d)
        mislabeled = df[df["labels"] != df["predictions"]]
        mislabeled["predictions"] = mislabeled["predictions"].astype(int)
        mislabeled.to_csv("./mislabeled.csv")
        print(confusion_matrix(labels_shuffle.flatten(), preds_concat.flatten()))

def param_sweep(params):
    for _ in range(params["num_sweeps"]):
        ret = params.copy()
        for k in params:
            if not k.endswith("_hparam"):
                continue
            hparam_spec = params[k]
            if hparam_spec["scale"] == "log":
                val = 10**np.random.uniform(*hparam_spec["range"])
            elif hparam_spec["scale"] == "linear":
                val = np.random.uniform(*hparam_spec["range"])
            ret[k.replace("_hparam", "")] = val
        yield ret

if __name__ == "__main__":
    params = params_util.load_params()
    tf.set_random_seed(230)
    with open("data/vocab.json") as f:
        vocab_json = json.load(f)
    vocab_list = vocab_as_sorted_list(vocab_json)
    vocab = tf.contrib.lookup.index_table_from_tensor(tf.constant(vocab_list), default_value=params["vocab_unk_idx"])

    glove_weights = np.load("data/glove.npy")

    tweets, lens = load_tweets_naive("data/dev/text.npy", params["max_len"])
    labels = load_labels_naive("data/dev/labels.npy")

    inputs = make_inputs(vocab, glove_weights, params["max_len"])
    train_model = model_fn("train", inputs, params)
    eval_model = model_fn("eval", inputs, params, reuse=True)

    if params.get("restore"):
        print(params)
        evaluate_model(inputs, params, eval_model, params["test_set_size"])