import numpy as np
import pickle
import tensorflow as tf


CLASS_QTY = 10
BATCH_SIZE = None
SGD_ENABLED = True


def load_mnist_data():
    with open('notMNIST.pickle', 'rb') as f:
        obj = pickle.load(f)
        print("Loaded notMNIST data...")
        return obj['samples'], obj['labels']


def divide_datasets(samples, labels, train_prop=0.75):
    n = samples.shape[0]
    perm = np.random.permutation(n)
    train_qty = int(train_prop * n)

    samples = samples[perm]
    labels = labels[perm]

    X_train, y_train = samples[:train_qty], labels[:train_qty]
    X_test, y_test = samples[train_qty:], labels[train_qty:]
    return X_train, y_train, X_test, y_test


def preprocess(training, testing):
    mean = np.mean(training, 0)
    std = np.std(training, 0)
    preprocess_f = lambda X: np.divide(X - mean, std)
    return preprocess_f(training), preprocess_f(testing), preprocess_f


def accuracy(prediction, labels):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).eval()


def get_batch(step, Xt_train, y_train, BATCH_SIZE, SGD_ENABLED=True):
    if SGD_ENABLED:
        offset = (step * BATCH_SIZE) % Xt_train.shape[0]
        samples = Xt_train[offset:offset + BATCH_SIZE, :]
        labels = y_train[offset:offset + BATCH_SIZE, :]
    else:
        samples = Xt_train
        labels = y_train

    perm = np.random.permutation(samples.shape[0])

    return samples[perm, :], labels[perm, :]

