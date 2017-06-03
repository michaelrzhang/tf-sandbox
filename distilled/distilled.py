
import os
import numpy as np
import tensorflow as tf
import pdb
from tensorflow.examples.tutorials.mnist import input_data
import csv

# which parts would be put in helper function?

# TODO: Better understand Tensorboard
# Visualizing the embeddings with t-SNE and PCA, along with your predicted labels seems very cool.
# Hyperparameter search can be done quite nicely.
# Figure out better ways to name variables (so that Tensorboard graph looks good) <- could use help
# Look at weight and bias activations.
# https://gist.github.com/dandelionmane/4f02ab8f1451e276fea1f165a20336f1 Dandelion Mion has a fantastic tutorial

# Try experiment omitting 3s
# shifts during training of first net

LOGDIR = '/tmp/mnist_hyperparams/'
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def nn(
    iterations = 5000,
    epochs = None,
    batch_size = 100,
    filters = 32,
    layers = [800, 800],
    output_size = 10,
    activation = tf.nn.relu,
    kernel_initializer=tf.contrib.layers.xavier_initializer(),
    learning_rate = 1e-3,
    regularizer = None,
    dropout = False,
    keep_probability = 0.5,
    conv_layers = None,
    exp_num = 0,
    use_pretrained_logits = False,
    model_name = "last",
    temperature = 5,
    soft_weight = 0.8,
    paper = False,
    ):
    if epochs is not None:
        iterations = int(60000 * epochs / batch_size)

    keep_prob = tf.placeholder(tf.float32, name = "kp")
    x = tf.placeholder(tf.float32, shape = [None, 784], name = "x")
    y_actual = tf.placeholder(tf.float32, shape = [None, 10], name = "y_actual")
    pretrained_logits = tf.placeholder(tf.float32, shape = [None, 10], name = "pretrained_logits")
    if use_pretrained_logits:
        train_logits = np.load("data/train_logits.npy")

    if conv_layers is not None:
        intermediate_layer = tf.reshape(x, [-1,28,28,1])
        for i in range(conv_layers):
            intermediate_layer = tf.contrib.layers.conv2d(
                inputs = intermediate_layer,
                num_outputs=filters,
                kernel_size=3,
                stride=1,
                weights_regularizer=regularizer,
                activation_fn=tf.nn.relu,
                padding = "SAME",)
        intermediate_layer = tf.reshape(intermediate_layer, [-1, filters * 28 * 28])
    else:
        intermediate_layer = x

    for layer_num, layer_size in enumerate(layers):
        intermediate_layer = tf.contrib.layers.fully_connected(
            inputs = intermediate_layer, 
            num_outputs = layer_size, 
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=regularizer,
            activation_fn = activation,
            )

        if dropout:
            intermediate_layer = tf.nn.dropout(intermediate_layer, keep_prob)
    # No nonlinearity on logits
    preactivation = tf.contrib.layers.fully_connected(
        inputs = intermediate_layer, 
        num_outputs = output_size, 
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=regularizer,
        # activation = tf.nn.relu,
        activation_fn =None,
        )

    if use_pretrained_logits:
        temp_one_preactivation = tf.identity(preactivation, name="logits")
        preactivation = tf.divide(preactivation, temperature)
        soft_labels = tf.nn.softmax(pretrained_logits /temperature)

        # follows paper
        if paper:
            # correction for temperature
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=preactivation, labels=soft_labels) * temperature ** 2

            # average of soft and hard
            cross_entropy = soft_weight * cross_entropy + \
             (1 - soft_weight) * tf.nn.softmax_cross_entropy_with_logits(logits=temp_one_preactivation, labels=y_actual) 

        else:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=preactivation, labels=soft_labels) 
            cross_entropy = soft_weight * cross_entropy + \
             (1 - soft_weight) * tf.nn.softmax_cross_entropy_with_logits(logits=preactivation, labels=y_actual)
    else:
        preactivation = tf.identity(preactivation, name="logits")
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=preactivation, labels=y_actual)
    mean_cross_entropy = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(mean_cross_entropy)
    correct_prediction = tf.equal(tf.argmax(preactivation, 1), tf.argmax(y_actual, 1)) # finds label
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = "accuracy") # reduce_mean requires float
    tf.summary.scalar('cross-entropy', mean_cross_entropy)
    tf.summary.scalar('accuracy', accuracy)

    summ = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init_op)

    # Creates a saver to save checkpoints
    saver = tf.train.Saver()

    print("Total iterations: {}".format(iterations))
    for i in range(iterations):
        random_indices = np.random.choice(55000, 100)
        feed_dict = feed_dict={x: mnist.train.images[random_indices], 
          y_actual: mnist.train.labels[random_indices], keep_prob:keep_probability}
        # add pretrained logits to feed dict if necessary
        if use_pretrained_logits:
            feed_dict[pretrained_logits] = train_logits[random_indices]
        # pdb.set_trace()
        sess.run(train_step, feed_dict)
        if i % 1000 == 0:
            acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y_actual:mnist.test.labels, keep_prob:1.0})
            print("Iteration: {} Training Data Seen: {} Validation Accuracy: {}".format(i, i * batch_size, acc))
            # saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
            
    val_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y_actual:mnist.test.labels, keep_prob:1.0}) 
    # train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y_actual:mnist.train.labels, keep_prob:1.0})
    train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images[:10000], y_actual:mnist.train.labels[:10000], keep_prob:1.0})
    print("Final validation accuracy: {}".format(val_acc))
    print("Final training accuracy: {}".format(train_acc))
    saver.save(sess, "models/" + model_name)
    return train_acc, val_acc

# struggling to load variables from checkpoint and to make predictions
def reload_model(save_logits = True):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('./models/bignet.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./models'))
        accuracy = sess.graph.get_tensor_by_name("accuracy:0")
        x = sess.graph.get_tensor_by_name("x:0")
        y_actual = sess.graph.get_tensor_by_name("y_actual:0")
        keep_prob = sess.graph.get_tensor_by_name("kp:0")
        logits = sess.graph.get_tensor_by_name("logits:0")
        accuracy = sess.graph.get_tensor_by_name("accuracy:0")
        predicted_logits, val_acc = sess.run([logits, accuracy], feed_dict={x:mnist.test.images, y_actual:mnist.test.labels, keep_prob:1.0}) 
        print("Final validation accuracy: {}".format(val_acc))
        
        if save_logits:    
            test_logits = sess.run(logits, feed_dict={x:mnist.test.images, y_actual:mnist.test.labels, keep_prob:1.0}) 
            train_logits = []
            np.save("data/test_logits", test_logits)
            # too big to fit in one batch
            for i in range(len(mnist.train.images) // 5000):
                # train_logits.append(sess.run(logits, feed_dict={x:mnist.train.images[i: i + 5000],  keep_prob:1.0}))
                acc, l = sess.run([accuracy, logits], feed_dict={x:mnist.train.images[i * 5000: (i + 1) * 5000], 
                 y_actual:mnist.train.labels[i * 5000: (i + 1) * 5000], keep_prob:1.0})
                print(acc)
                train_logits.append(l)
            # 55000 samples because tensorflow uses a validation set
            train_logits = np.concatenate(train_logits, axis = 0)
            np.save("data/train_logits", train_logits)
        validate_logits()
         

def validate_logits():      
    train_logits = np.load("data/train_logits.npy")
    correct = (np.argmax(train_logits, axis = 1) == np.argmax(mnist.train.labels, axis = 1))
    print("Correct: {}, Total: {}".format(np.sum(correct), len(correct)))

    test_logits = np.load("data/test_logits.npy")
    correct = (np.argmax(test_logits, axis = 1) == np.argmax(mnist.test.labels, axis = 1))
    print("Validation Correct: {}, Total: {}".format(np.sum(correct), len(correct)))

def hyperparam_search():
    data = []
    learning_rate = 1e-3
    for temperature in [20]:
        for soft_weight in [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]:
            for paper in [True, False]:
                soft_train, soft_val = nn(layers = [800, 800],
                    learning_rate = learning_rate,
                    epochs = 30, 
                    # exp_num = 3,
                    use_pretrained_logits = True,
                    temperature = temperature,
                    soft_weight = soft_weight,
                    paper=paper,
                    )
                data.append((paper, temperature, soft_weight, soft_train, soft_val))
            # break
    for row in data:
        print(row)

    out = "temperature_weight_hyperparamsearch4.csv"
    with open(out, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(("paper", "temperature", "soft_weight", "soft_train", "soft_val"))
        for row in data:
            writer.writerow(row)


    # hyperparamsearc2 using the cumbersome model with a high temperature in its softmax.

def train():
    learning_rate = 1e-3

    # nn(layers = [800, 800],
    #     learning_rate = learning_rate,
    #     epochs = 8, 
    #     dropout = True, 
    #     keep_probability = 0.5,
    #     regularizer=tf.contrib.layers.l2_regularizer(1e-2),
    #     conv_layers = 5,
    #     filters = 32,
    #     exp_num = 3,
    #     use_pretrained_logits = False,
    #     )
    soft_train, soft_val = nn(layers = [800, 800],
        learning_rate = learning_rate,
        epochs = 50, 
        use_pretrained_logits = True,
        temperature = 20,
        soft_weight = 0.99,
        paper = False,
        )

    hard_train, hard_val = nn(layers = [800, 800],
        learning_rate = learning_rate,
        epochs = 50, 
        use_pretrained_logits = False,
        )
    
    print("Hard train: {} Hard val: {} Soft train: {} Soft val: {}".format(hard_train, hard_val, soft_train, soft_val))

# soft: 0.95 Hard train: 0.9991002082824707 Hard val: 0.9836001396179199 Soft train: 0.999900221824646 Soft val: 0.9881001710891724


def main():
    # validate_logits()
    train()
    # reload_model()
    # hyperparam_search()

if __name__ == '__main__':
    #argparse
    main()

# 3 epochs of training [400, 400] 4 layers
# Final validation accuracy: 0.989500105381012
# Final training accuracy: 0.9963001012802124

# 3 epochs of training [400, 400] 5 layers
# Final validation accuracy: 0.9899001121520996
# Final training accuracy: 0.994300127029419

# 3 epochs of training [200, 200] 6 layers
# Final validation accuracy: 0.9899001121520996
# Final training accuracy: 0.994300127029419

# best model gets 9930