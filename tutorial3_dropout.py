import tensorflow as tf
from utils import *


DROPOUT_PROB = 0.3
STEP_QTY = 1000
LAYER_SIZE = 1024
FEATURE_QTY = 784
layer_comp = {
    0: FEATURE_QTY,
    1: LAYER_SIZE,
    2: CLASS_QTY,
}
LAYERS = len(layer_comp)
TRAINING_STEP = 1 / (10 ** LAYERS)


samples, labels = load_mnist_data()
X_train, y_train, X_test, y_test = divide_datasets(samples, labels)
Xt_train, Xt_test, _ = preprocess(X_train, X_test)

BATCH_SIZE = int(X_train.shape[0] / 100)
print("Batch size is: ", BATCH_SIZE)

weights = {}
biases = {}
result = {}
test_result = {}

graph = tf.Graph()
with graph.as_default():
    predictors = tf.placeholder(tf.float32, shape=(None, FEATURE_QTY))
    targets = tf.placeholder(tf.float32, shape=(None, CLASS_QTY))

    test_predictors = tf.constant(Xt_test, dtype=tf.float32)
    test_targets = tf.constant(y_test, dtype=tf.float32)

    result[0] = predictors
    test_result[0] = test_predictors

    for layer in range(1, LAYERS):
        shape = (layer_comp[layer - 1], layer_comp[layer])
        bias_shape = (layer_comp[layer],)

        weights[layer] = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32))
        biases[layer] = tf.Variable(tf.zeros(bias_shape, dtype=tf.float32))

    for layer in range(1, LAYERS):
        W = weights[layer]
        b = biases[layer]

        result[layer] = tf.nn.dropout(tf.matmul(result[layer - 1], W) + b, DROPOUT_PROB)
        test_result[layer] = tf.matmul(test_result[layer - 1], W) + b

        if layer < LAYERS - 1:
            result[layer] = tf.nn.relu(result[layer])
            test_result[layer] = tf.nn.relu(test_result[layer])

    reg_term = None
    for w in weights.values():
        if reg_term is None:
            reg_term = tf.nn.l2_loss(w)
        else:
            reg_term += tf.nn.l2_loss(w)

    loss_term = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(result[LAYERS - 1], targets))

    total_loss = loss_term + reg_term
    optimizer = tf.train.GradientDescentOptimizer(TRAINING_STEP).minimize(total_loss)

    output = tf.nn.softmax(result[LAYERS - 1])
    test_output = tf.nn.softmax(test_result[LAYERS - 1])


print("Finished creating computation graph...")


with tf.Session(graph=graph) as session:
    session.run(tf.initialize_all_variables())

    for step in range(STEP_QTY):
        samples, labels = get_batch(step, Xt_train, y_train, BATCH_SIZE, SGD_ENABLED)

        feed_dict = {predictors: samples, targets: labels}

        if step % 20 == 0:
            _, l, pred = session.run([optimizer, total_loss, test_output], feed_dict=feed_dict)
            print("Step %d:\n\tloss %f\n\tbatch err %f" % (step, l, accuracy(pred, y_test)))
        else:
            session.run([optimizer], feed_dict=feed_dict)
