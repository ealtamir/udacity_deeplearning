import glob
import numpy as np
import pickle
import scipy.misc as misc
import tensorflow as tf


BATCH_SIZE = None
CLASS_QTY = 10
GLOB_DATASET_PATH = '../datasets/notMNIST_large/*/*.png'
LEARNING_RATE = 0.01
PICKLE_FILE = 'notMNIST.pickle'
SGD_ENABLED = True
TEST_SET_PROPORTION = 0.25
TRAINING_STEPS = 500
MAX_SAMPLES = 100000
IMAGE_SIZE = 28
HIDDEN_NODES = 1024
REG_TERM = 0.001


def load_pickle_dataset():
    global BATCH_SIZE
    with open(PICKLE_FILE, 'rb') as f:
        obj = pickle.load(f)
    s, t = obj['samples'], obj['labels']
    BATCH_SIZE = int(s.shape[0] / 40)
    print("Batch size: ", BATCH_SIZE)
    print("Loaded pickle dataset...")
    return s, t


def divide_dataset(samples, targets):
    assert samples.shape[0] == targets.shape[0]
    sample_qty = samples.shape[0]
    train_size = int(sample_qty * (1 - TEST_SET_PROPORTION))
    return samples[:train_size], targets[:train_size], \
           samples[train_size:], targets[train_size:]


def reformat(targets):
    row_qty = targets.shape[0]
    new_targets = np.zeros((row_qty, CLASS_QTY))
    new_targets[np.arange(row_qty), targets.reshape(1, row_qty)] = 1
    return new_targets


def accuracy(prediction, labels):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).eval()


def load_data():
    file_list = glob.glob(GLOB_DATASET_PATH)
    file_qty = min(len(file_list), MAX_SAMPLES)
    perm = np.random.permutation(file_qty)
    samples = None
    labels = None
    for i in range(file_qty):
        path = file_list[perm[i]]
        if i % 1000 == 0:
            print('processing file %d' % i)
        label = ord(path.split('/')[3]) - ord('A')
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


#samples, targets = load_data()
#save_to_pickle(samples, targets)

samples, targets = load_pickle_dataset()
X_train, y_train, X_test, y_test = divide_dataset(samples, targets)

BATCH_SIZE = X_train.shape[0] / 100

X_mean = np.mean(X_train, 0)
X_std = np.std(X_train, 0)
preprocess = lambda X: np.divide(X - X_mean, X_std)

X_train = preprocess(X_train)
X_test = preprocess(X_test)


graph = tf.Graph()
with graph.as_default():
    training_batch = tf.placeholder(tf.float32, shape=(None, 784))
    label_batch = tf.placeholder(tf.float32, shape=(None, CLASS_QTY))
    test_batch = tf.constant(X_test, dtype=tf.float32)

    #W = tf.Variable(tf.truncated_normal((784, CLASS_QTY)))
    #bias = tf.Variable(tf.zeros([CLASS_QTY]))

    #logits = tf.matmul(training_batch, W) + bias
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, label_batch))

    #optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    #train_prediction = tf.nn.softmax(logits)
    #test_prediction = tf.nn.softmax(tf.matmul(test_batch, W) + bias)

    w1 = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, HIDDEN_NODES]), name="w1")
    b1 = tf.Variable(tf.zeros([HIDDEN_NODES]), name="b1")

    w2 = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, HIDDEN_NODES]), name="w2")
    b2 = tf.Variable(tf.zeros([HIDDEN_NODES]), name="b2")

    w3 = tf.Variable(tf.truncated_normal([HIDDEN_NODES, CLASS_QTY]), name="w3")
    b3 = tf.Variable(tf.zeros([CLASS_QTY]), name="b3")

    layer1 = tf.matmul(training_batch, w1) + b1
    activation_layer1 = tf.nn.relu(layer1)

    layer2 = tf.matmul(training_batch, w2) + b2
    activation_layer2 = tf.nn.relu(layer2)

    layer3 = tf.matmul(activation_layer2, w3) + b3
    regularization_loss = REG_TERM * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))

    weight_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(layer3, label_batch))
    loss = weight_loss + regularization_loss

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    train_prediction = tf.nn.softmax(layer2)
    r1 = tf.nn.relu(tf.matmul(test_batch, w1) + b1)
    r2 = tf.nn.relu(tf.matmul(test_batch, w2) + b2)
    test_prediction = tf.nn.softmax(tf.matmul(r2, w3) + b3)

with tf.Session(graph=graph) as session:
    session.run(tf.initialize_all_variables())

    for step in range(TRAINING_STEPS):
        if SGD_ENABLED:
            offset = (step * BATCH_SIZE) % X_train.shape[0]
            samples = X_train[offset:offset + BATCH_SIZE, :]
            labels = y_train[offset:offset + BATCH_SIZE, :]
        else:
            samples = X_train
            labels = y_train

        perm = np.random.permutation(samples.shape[0])

        samples = samples[perm, :]
        labels = labels[perm, :]

        feed_dict = {training_batch: samples, label_batch: labels}

        if step % 100 == 0:
            _, l, pred = session.run([optimizer, loss, test_prediction], feed_dict=feed_dict)
            print("Step %d:\n\tloss %f\n\tbatch err %f" % (step, l, accuracy(pred, y_test)))
        else:
            session.run([optimizer], feed_dict=feed_dict)

