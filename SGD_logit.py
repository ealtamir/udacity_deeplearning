import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pickle

from scipy import misc


CLASS_QTY = 10
TRAINING_STEPS = 1000
LEARNING_RATE = 0.1

np.random.seed(0)

def extract_label(key):
    parts = key.decode('ASCII').split('/')
    label = ord(parts[2]) - ord('A')
    one_hot_label = [0] * CLASS_QTY
    one_hot_label[label] = 1
    return np.asarray(one_hot_label).reshape(1, 10)


sample_files = glob.glob('./notMNIST_small/*/*.png')
file_queue = tf.train.string_input_producer(sample_files, shuffle=True)
image_reader = tf.WholeFileReader()


W = tf.Variable(tf.random_normal([784, 10], stddev=1))
b = tf.Variable(tf.zeros([10]))

X = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    yhat = tf.nn.softmax(tf.matmul(X, W) + b)
    neg_log = -tf.reduce_sum(y * tf.log(tf.clip_by_value(yhat, 1e-10, 1.0)))
    ce_loss = tf.reduce_mean(neg_log)

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(ce_loss)

    for step in range(TRAINING_STEPS):
        key, image_file = image_reader.read(file_queue)
        image = tf.reshape(tf.image.decode_png(image_file, channels=1), [-1]).eval().reshape(1, 784)
        label = extract_label(key.eval())

        _, l = sess.run([optimizer, ce_loss], feed_dict={X: image, y: label})
        print("%d) Loss: %f" % (step, l))


#    image_tensor = image.eval()
    coord.request_stop()
    coord.join(threads)

