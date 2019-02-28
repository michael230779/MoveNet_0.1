from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import glob
import os

tf.enable_eager_execution()

files = glob.glob(os.path.join("..", "MoveNet_0.1", "data", "*", "*.csv"))

# reading in the x,y,z features
test_running_features = np.genfromtxt(files[0], delimiter=',', skip_header=1, usecols=(1, 2, 3))
test_walking_features = np.genfromtxt(files[1], delimiter=',', skip_header=1, usecols=(1, 2, 3))
train_running_features = np.genfromtxt(files[2], delimiter=',', skip_header=1, usecols=(1, 2, 3))
train_walking_features = np.genfromtxt(files[3], delimiter=',', skip_header=1, usecols=(1, 2, 3))

# reshaping features so that one sample consists of 10 data points (resembling 2s at around 5 Hz)
test_running_features = np.reshape(test_running_features, [886, 30])
test_walking_features = np.reshape(test_walking_features, [883, 30])
train_running_features = np.reshape(train_running_features, [3548, 30])
train_walking_features = np.reshape(train_walking_features, [3537, 30])

train_features = np.concatenate([train_running_features, train_walking_features], axis=0)
test_features = np.concatenate([test_running_features, test_walking_features], axis=0)

# creating labels in corresponding shape
test_running_labels = np.ones([886, 1], dtype=np.int32)
test_walking_labels = np.zeros([883, 1], dtype=np.int32)
train_running_labels = np.ones([3548, 1], dtype=np.int32)
train_walking_labels = np.zeros([3537, 1], dtype=np.int32)

train_labels = np.concatenate([train_running_labels, train_walking_labels], axis=0)
test_labels = np.concatenate([test_running_labels, test_walking_labels], axis=0)


# write input functions which need to yield a (feature dict, labels) to allow for use of Estimator
def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(({"features": train_features}, train_labels))

    # shuffle and repeat data, for count=None data is repeated indefinitely, shuffle buffer_size is set to lenght of
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(7085, count=None))

    # batch dataset to batches of 25 samples
    dataset = dataset.batch(25)

    return dataset


def test_input_fn():
    test_dataset = tf.data.Dataset.from_tensor_slices(({"features": test_features}, test_labels))

    # shuffle and repeat data, for count=None data is repeated indefinitely, shuffle buffer_size is set to lenght of dataset
    test_dataset = test_dataset.apply(tf.data.experimental.shuffle_and_repeat(1769, count=None))

    # batch dataset to batches of 25 samples
    test_dataset = test_dataset.batch(25)

    return test_dataset


def model_fn(features, labels, mode, params):

    net = tf.feature_column.input_layer(features, params['feature_columns'])

    net = tf.identity(net, name="input_tensor")

    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # apply dropout but only in training mode, seed is necessary to allow for tflite conversion
    # net = tf.layers.dropout(net, rate=0.2, seed=1, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # handle prediction mode
    predicted_class = tf.argmax(logits, 1, name="prediction")
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"class_ids": predicted_class[:, tf.newaxis],
                       "probabilities": tf.nn.softmax(logits), "logits": logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    # compute evaluation metrics, REVIEW: consider adding/removing metrics
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_class, name="acc_op")
    precision = tf.metrics.precision(labels=labels, predictions=predicted_class, name="prec_op")
    recall = tf.metrics.recall(labels=labels, predictions=predicted_class, name="rec_op")
    false_negatives = tf.metrics.false_negatives(labels=labels, predictions=predicted_class, name="fn_op")
    true_negatives = tf.metrics.true_negatives(labels=labels, predictions=predicted_class, name="tn_op")
    false_positives = tf.metrics.false_positives(labels=labels, predictions=predicted_class, name="fp_op")
    true_positives = tf.metrics.true_positives(labels=labels, predictions=predicted_class, name="tp_op")
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "false_negatives": false_negatives,
               "true_negatives": true_negatives, "false_positives": false_positives, "true_positives": true_positives}

    # make evaluation metrics available to TensorBoard both for training and evaluation mode, loss is added automatically
    for key in metrics:
        tf.summary.scalar(str(key), metrics[key][1])

    # handle evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # handle training mode, REVIEW: consider changing optimizer and learning_rate
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


features = tf.feature_column.numeric_column("features", shape=[30])

# save_summary_steps every 5 and save checkpoints every 900 sec
params = {"feature_columns": [features], "n_classes": 2, "learning_rate": 0.001, "hidden_units": [40, 20, 20]}

estimator = tf.estimator.Estimator(config=tf.estimator.RunConfig(
    model_dir=os.path.join("model_logs", "2019_02_27_1832"), save_summary_steps=5, save_checkpoints_secs=2), model_fn=model_fn, params=params)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=2000)
eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn, steps=35, throttle_secs=2, start_delay_secs=0.5)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# iterating through dataset for debugging purposes
test_train_dataset = train_input_fn()

iterator = test_train_dataset.make_one_shot_iterator()

next_element = iterator.get_next()
print(next_element)
