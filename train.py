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

def train_loop(inputs, params, train_model, eval_model):
    batch_size = params["batch_size"]
    train_set_size = params["train_set_size"]
    dev_set_size = params["dev_set_size"]
    save_dir = os.path.join(os.getcwd(), "results", params["run_id"])
    save_path = os.path.join(save_dir, "after-epoch")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with tf.Session() as sess:
        sess.run(train_model['variable_init_op'])

        saver = tf.train.Saver()
        if params.get("restore"):
            saver.restore(sess, tf.train.latest_checkpoint(save_dir))
        num_minibatches =  math.ceil(train_set_size / float(batch_size))
        global_step = tf.train.get_global_step()
        start_epoch = params.get("last_epoch", -1) + 1

        train_writer = tf.summary.FileWriter("./logs/"+params["run_id"]+"_train")
        dev_writer = tf.summary.FileWriter("./logs/"+params["run_id"]+"_dev")

        for epoch in range(start_epoch, start_epoch + params["num_epochs"]):
            print("=== Epoch %d ===" % epoch)
            params["last_epoch"] = epoch
            tweets_shuffle, lens_shuffle, labels_shuffle = shuffle_datasets([tweets, lens, labels])

            # Train the model
            print("Training...")
            train_writer = tf.summary.FileWriter("./logs/train")
            sess.run(train_model['metrics_init_op'])
            t = trange(num_minibatches)
            for batch_num in t:
                start_idx = batch_num*batch_size
                end_idx = min((batch_num+1)*batch_size, train_set_size)
                feed_dict = {
                    inputs["tweets"]: tweets_shuffle[start_idx:end_idx],
                    inputs["lengths"]: lens_shuffle[start_idx:end_idx],
                    inputs["labels"]: labels_shuffle[start_idx:end_idx],
                    inputs["keep_prob"]: params["dropout_keep_prob"],
                    inputs["l2_lambda"]: params["l2_lambda"]
                }
                _, _, summary, loss, accuracy = sess.run(
                    [train_model["train_op"],
                    train_model["update_metrics"],
                    train_model["summary_op"],
                    train_model["loss"],
                    train_model["accuracy"]], feed_dict)
                global_step_val = sess.run(global_step)

                if batch_num % 100 == 0:
                    train_writer.add_summary(summary, global_step_val)
                    train_writer.flush()
                t.set_postfix(loss='{:05.3f}'.format(loss), accuracy='{:05.3f}'.format(accuracy))
            train_metric_values = sess.run({k: v[0] for k, v in train_model["metrics"].items()})
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metric_values.items())
            print(metrics_string)

            # Evaluate on dev set
            print("Evaluating train set...")
            sess.run(eval_model['metrics_init_op'])
            tweets_shuffle, lens_shuffle, labels_shuffle = shuffle_datasets([tweets, lens, labels])
            for batch_num in range(num_minibatches):
                start_idx = batch_num*batch_size
                end_idx = min((batch_num+1)*batch_size, dev_set_size)
                feed_dict = {
                    inputs["tweets"]: tweets_shuffle[start_idx:end_idx],
                    inputs["lengths"]: lens_shuffle[start_idx:end_idx],
                    inputs["labels"]: labels_shuffle[start_idx:end_idx],
                    inputs["keep_prob"]: 1.0,
                    inputs["l2_lambda"]: params["l2_lambda"]
                }
                sess.run(eval_model["update_metrics"], feed_dict)
            dev_metric_values = sess.run({k: v[0] for k, v in eval_model["metrics"].items()})
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in dev_metric_values.items())
            print(metrics_string)

            # Evaluate on dev set
            print("Evaluating dev set...")
            dev_minibatches = math.ceil(dev_set_size / float(batch_size))
            sess.run(eval_model['metrics_init_op'])
            tweets_shuffle, lens_shuffle, labels_shuffle = shuffle_datasets([dev_tweets, dev_lens, dev_labels])
            for batch_num in range(dev_minibatches):
                start_idx = batch_num*batch_size
                end_idx = min((batch_num+1)*batch_size, dev_set_size)
                feed_dict = {
                    inputs["tweets"]: tweets_shuffle[start_idx:end_idx],
                    inputs["lengths"]: lens_shuffle[start_idx:end_idx],
                    inputs["labels"]: labels_shuffle[start_idx:end_idx],
                    inputs["keep_prob"]: 1.0,
                    inputs["l2_lambda"]: params["l2_lambda"]
                }
                sess.run(eval_model["update_metrics"], feed_dict)
                global_step_val = sess.run(global_step)
            dev_metric_values = sess.run({k: v[0] for k, v in eval_model["metrics"].items()})
            for tag, val in dev_metric_values.items():
                    summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
                    dev_writer.add_summary(summ, global_step_val)
                    dev_writer.flush()
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in dev_metric_values.items())
            print(metrics_string)
            saver.save(sess, save_path, global_step=epoch)

            with open(save_dir + "/results.json", "w") as f:
                results = {
                    "train_metrics": train_metric_values,
                    "dev_metrics": dev_metric_values
                }
                json.dump(results, f, cls=MyEncoder, indent=4, separators=(',', ': '))
            with open(save_dir + "/params.json", "w") as f:
                json.dump(params, f, cls=MyEncoder, indent=4, separators=(',', ': '))

        return train_metric_values, dev_metric_values

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

    tweets, lens = load_tweets_naive("data/train/text.npy", params["max_len"])
    labels = load_labels_naive("data/train/labels.npy")

    dev_tweets, dev_lens = load_tweets_naive("data/dev/text.npy", params["max_len"])
    dev_labels = load_labels_naive("data/dev/labels.npy")

    inputs = make_inputs(vocab, glove_weights, params["max_len"])
    train_model = model_fn("train", inputs, params)
    eval_model = model_fn("eval", inputs, params, reuse=True)

    if params.get("restore"):
        print(params)
        train_metrics, dev_metrics = train_loop(inputs, params, train_model, eval_model)
    else:
        for p in param_sweep(params):
            ts = int(datetime.datetime.utcnow().timestamp())
            p["run_id"] = "%s_%d" % (params["experiment"], ts)
            print(p)
            train_metrics, dev_metrics = train_loop(inputs, p, train_model, eval_model)

