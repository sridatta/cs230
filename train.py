import tensorflow as tf
import numpy as np
from model.input_fn import input_fn
from model.input_fn import load_labels
from model.input_fn import load_tweets, load_tweets_naive, load_labels_naive, make_inputs
from generate_vocab import vocab_as_sorted_list
import json
import params as params_util
from model.model_fn import model_fn

def shuffle_datasets(datasets):
    shuffle_idx = np.random.permutation(len(datasets[0]))
    return tuple(dataset[shuffle_idx] for dataset in datasets)

if __name__ == "__main__":
    params = params_util.load_params()
    print(params)
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

    batch_size = params["batch_size"]
    with tf.Session() as sess:
        sess.run(train_model['variable_init_op'])
        num_minibatches = params["train_set_size"] // params["batch_size"]
        writer = tf.summary.FileWriter("logs", sess.graph)
        global_step = tf.train.get_global_step()
        for _ in range(params["num_epochs"]):
            tweets_shuffle, lens_shuffle, labels_shuffle = shuffle_datasets([tweets, lens, labels])

            # Train the model
            sess.run(train_model['metrics_init_op'])
            for batch_num in range(num_minibatches):
                feed_dict = {
                    inputs["tweets"]: tweets_shuffle[batch_num*batch_size:(batch_num+1)*batch_size],
                    inputs["lengths"]: lens_shuffle[batch_num*batch_size:(batch_num+1)*batch_size],
                    inputs["labels"]: labels_shuffle[batch_num*batch_size:(batch_num+1)*batch_size]
                }
                _, loss, accuracy = sess.run(
                    [train_model["train_op"],
                    train_model["loss"],
                    train_model["accuracy"]], feed_dict)
                if batch_num % 1000 == 0:
                    print("Minibatch Train Loss: %f" % loss)
                    print("Minibatch Train Accuracy: %f" % accuracy)

            # Evaluate on dev set
            dev_minibatches = params["dev_set_size"] // params["batch_size"]
            sess.run(eval_model['metrics_init_op'])
            tweets_shuffle, lens_shuffle, labels_shuffle = shuffle_datasets([dev_tweets, dev_lens, dev_labels])
            for batch_num in range(dev_minibatches):
                feed_dict = {
                    inputs["tweets"]: tweets_shuffle[batch_num*batch_size:(batch_num+1)*batch_size],
                    inputs["lengths"]: lens_shuffle[batch_num*batch_size:(batch_num+1)*batch_size],
                    inputs["labels"]: labels_shuffle[batch_num*batch_size:(batch_num+1)*batch_size]
                }
                sess.run(eval_model["update_metrics"], feed_dict)
            metric_values = sess.run({k: v[0] for k, v in eval_model["metrics"].items()})
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metric_values.items())
            print(metrics_string)
        writer.close()