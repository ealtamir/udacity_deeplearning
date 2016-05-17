import os
import glob

import scipy.ndimage as misc
import numpy as np
import pickle
import tensorflow as tf


CLASS_QTY = 10
BATCH_SIZE = None
SGD_ENABLED = True
GLOB_DATASET_PATH = '/Users/work/Documents/datasets/notMNIST_large/*/*.png'
MAX_SAMPLES = 1000000
PICKLE_FILE = 'notMNIST.pickle'


def load_data_from_pickle_file():
    with open('notMNIST.pickle', 'rb') as f:
        obj = pickle.load(f)
        print("Loaded notMNIST data...")
        return obj['samples'], obj['labels']


def load_data_from_dataset_dir():
    file_list = glob.glob(GLOB_DATASET_PATH)
    file_qty = min(len(file_list), MAX_SAMPLES)
    perm = np.random.permutation(file_qty)
    samples = None
    labels = None
    for i in range(file_qty):
        path = file_list[perm[i]]
        if i % 1000 == 0:
            print('processing file %d' % i)
        label = ord(path.split('/')[-2]) - ord('A')
        try:
            arr = misc.imread(path).reshape(1, -1)
        except:
            continue
        label_vec = np.zeros((1, CLASS_QTY))
        label_vec[:, label] = 1
        if samples is None:
            samples = np.matrix(arr)
            labels = np.matrix(label_vec)
        else:
            samples = np.vstack((samples, arr))
            labels = np.vstack((labels, label_vec))
    return samples, labels


def save_to_pickle(samples, labels):
    with open(PICKLE_FILE, 'wb') as f:
        obj = {'samples': samples, 'labels': labels}
        pickle.dump(obj, f)
        print("file saved to pickle")


def load_pickle_dataset():
    global BATCH_SIZE
    with open(PICKLE_FILE, 'rb') as f:
        obj = pickle.load(f)
    s, t = obj['samples'], obj['labels']
    BATCH_SIZE = int(s.shape[0] / 40)
    print("Batch size: ", BATCH_SIZE)
    print("Loaded pickle dataset...")
    return s, t


def load_mnist_data():
    if os.path.exists('notMNIST.pickle'):
        return load_data_from_pickle_file()

    samples, labels = load_data_from_dataset_dir()
    save_to_pickle(samples, labels)
    return samples, labels


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

