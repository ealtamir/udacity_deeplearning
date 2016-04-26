import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pickle

from scipy import misc


np.random.seed(0)


SAMPLE_QTY = 30000
TRAINING_STEP = 0.001
TRAINING_EPOCHS = 1000
CLASS_QTY = None
TEST_SET_PROPORTION = 0.25
PICKLE_FILENAME = 'notMNIST.pickle'


def load_samples(filenames):
    matrix = None
    for filename in filenames:
        try:
            arr = misc.imread(filename)
        except:
            continue
        length = arr.shape[0] * arr.shape[1]
        sample = arr.reshape(1, length)
        if matrix is not None:
            matrix = np.vstack((matrix, sample))
        else:
            matrix = np.matrix(sample)
    return matrix


def load_dataset():
    global CLASS_QTY
    classes = glob.glob('../datasets/notMNIST_large/*')
    sample_files = {}
    for class_filepath in classes:
        name = class_filepath[-1]
        sample_files[name] = np.asarray(glob.glob('%s/*.png' % class_filepath))

    keys = tuple(sample_files.keys())
    CLASS_QTY = len(keys)
    class_sample_qty = SAMPLE_QTY // CLASS_QTY

    X = None
    y = None
    for key, val in sample_files.items():
        cls_value_qty = len(val)
        limit = min(cls_value_qty, class_sample_qty)
        perm = np.random.permutation(cls_value_qty)[:limit]
        samples = val[perm]
        mat = load_samples(samples)
        cls_id = ord(key) - ord('A')
        targets = np.zeros((mat.shape[0], CLASS_QTY))
        targets[:, cls_id] = 1
        if X is not None:
            X = np.vstack((X, mat))
            y = np.vstack((y, targets))
        else:
            X = mat
            y = targets
    perm = np.random.permutation(X.shape[0])
    X = X[perm, :]
    y = y[perm]
    return X, y


def divide_dataset(samples, targets):
    assert samples.shape[0] == targets.shape[0]
    sample_qty = samples.shape[0]
    train_size = int(sample_qty * (1 - TEST_SET_PROPORTION))
    return samples[:train_size], targets[:train_size],\
           samples[train_size:], targets[train_size:]


def manual_testing(samples, targets, predictions):
    sample_qty = samples.shape[0]
    for i in range(20):
        index = np.random.randint(0, sample_qty)
        arr = samples[index, :].reshape(28, 28)
        tar = targets[index]
        pred = predictions[index]
        print("%d) Sample %d: is %s, but %s was predicted" % (i, index, chr(tar + ord('A')), chr(pred + ord('A'))))
        plt.imshow(arr, cmap='Greys_r')
        plt.show()


if __name__ == '__main__':
    if os.path.exists(PICKLE_FILENAME):
        with open(PICKLE_FILENAME, 'rb') as f:
            obj = pickle.load(f)
            samples, targets = obj['samples'], obj['targets']
            print("Dataset loaded from file...")
    else:
        samples, targets = load_dataset()
        with open(PICKLE_FILENAME, 'wb') as f:
            obj = {'samples': samples, 'targets': targets}
            pickle.dump(obj, f)
            print("File saved....")

    X_train, y_train, X_test, y_test = divide_dataset(samples, targets)

    train_mean = np.mean(X_train, 0)
    train_std = np.std(X_train, 0)
    preprocess = lambda X : np.divide(X - train_mean, train_std)

    X_proc = preprocess(X_train)
    X_test = preprocess(X_test)


    X = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    W = tf.Variable(tf.random_normal([784, 10], stddev=1))
    b = tf.Variable(tf.zeros([10]))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        yhat = tf.nn.softmax(tf.matmul(X, W) + b)

        #neg_log = -tf.reduce_sum(y * tf.log(yhat), reduction_indices=[1])
        neg_log = -tf.reduce_sum(y * tf.log(tf.clip_by_value(yhat, 1e-10, 1.0)))
        ce_loss = tf.reduce_mean(neg_log)

        train_step = tf.train.GradientDescentOptimizer(TRAINING_STEP).minimize(ce_loss)

        for iter in range(TRAINING_EPOCHS):
            train_step.run(feed_dict={X: X_proc, y: y_train})
            if iter % 100 == 0:
                print("Iter %d: %f" % (iter, ce_loss.eval({X: X_proc, y: y_train})))

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval({X: X_train, y: y_train}))
        real = tf.argmax(y, 1).eval({X: X_train, y: y_train})
        pred = tf.argmax(yhat, 1).eval({X: X_train, y: y_train})
        manual_testing(X_train, real, pred)

