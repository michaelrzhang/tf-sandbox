
import os
import numpy as np
import tensorflow as tf
import pdb
from tensorflow.examples.tutorials.mnist import input_data

# which parts would be put in helper function?

# TODO: Better understand Tensorboard
# Visualizing the embeddings with t-SNE and PCA, along with your predicted labels seems very cool.
# Hyperparameter search can be done quite nicely.
# Figure out better ways to name variables (so that Tensorboard graph looks good) <- could use help
# Look at weight and bias activations.
# https://gist.github.com/dandelionmane/4f02ab8f1451e276fea1f165a20336f1 Dandelion Mion has a fantastic tutorial

LOGDIR = '/tmp/mnist_hyperparams/'
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def nn(
    iterations = 5000,
    epochs = None,
    batch_size = 100,
    layers = [800, 800],
    output_size = 10,
    activation = tf.nn.relu,
    kernel_initializer=tf.contrib.layers.xavier_initializer(),
    learning_rate = 1e-3,
    regularizer = None,
    dropout = False,
    keep_probability = 1.0,
    conv_layers = None,
    exp_num = 0,
    ):
    if epochs is not None:
        iterations = 60000 * epochs // batch_size

    keep_prob = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, shape = [None, 784])
    y_actual = tf.placeholder(tf.float32, shape = [None, 10])

    # kinda hacky
    if conv_layers is not None:
        intermediate_layer = tf.reshape(x, [-1,28,28,1])
        for i in range(conv_layers):
            intermediate_layer = tf.contrib.layers.conv2d(
                inputs = intermediate_layer,
                num_outputs=32,
                kernel_size=3,
                stride=1,
                weights_regularizer=regularizer,
                activation_fn=tf.nn.relu,
                padding = "SAME",)
        intermediate_layer = tf.reshape(intermediate_layer, [-1, 32 * 28 * 28])
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

    if dropout:
        preactivation = tf.nn.dropout(preactivation, keep_prob + 0.3) # <- TODO:this is bad 80%

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=preactivation, labels=y_actual)
    mean_cross_entropy = tf.reduce_mean(cross_entropy)
    #train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(mean_cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(mean_cross_entropy)
    correct_prediction = tf.equal(tf.argmax(preactivation, 1), tf.argmax(y_actual, 1)) # finds label
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # reduce_mean requires float
    tf.summary.scalar('cross-entropy', mean_cross_entropy)
    tf.summary.scalar('accuracy', accuracy)

    summ = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()
    sess = tf.Session()
   

    # Creates a writer to write summaries
    writer = tf.summary.FileWriter(LOGDIR + str(exp_num))
    writer.add_graph(sess.graph)
    sess.run(init_op)

#    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
#    embedding_config = config.embeddings.add()
#    embedding_config.tensor_name = embedding.name
#    embedding_config.sprite.image_path = LOGDIR + 'sprite_1024.png'
#    embedding_config.metadata_path = LOGDIR + 'labels_1024.tsv'
#    # Specify the width and height of a single thumbnail.
#    embedding_config.sprite.single_image_dim.extend([28, 28])
#    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    # Creates a saver to save checkpoints
    saver = tf.train.Saver()

    print("Total iterations: {}".format(iterations))
    for i in range(iterations):
        batch = mnist.train.next_batch(batch_size)
        # try:
            s,  _ = sess.run([summ, train_step], feed_dict={x: batch[0], y_actual: batch[1], keep_prob:keep_probability})
        # except Exception as e:
        #     import pdb; pdb.set_trace()
        if i % 5 == 0:
            # print(keep_probability)
            # print(type(keep_probability))
            # s = sess.run(summ, feed_dict={x: batch[0], y_actual: batch[1], keep_prob:keep_probability})
            writer.add_summary(s, i)
        if i % 1000 == 0:
            acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y_actual:mnist.test.labels, keep_prob:1.0})
            print("Iteration: {} Training Data Seen: {} Validation Accuracy: {}".format(i, i * batch_size, acc))
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
            
    val_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y_actual:mnist.test.labels, keep_prob:1.0}) 
    # train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y_actual:mnist.train.labels, keep_prob:1.0})
    train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images[:10000], y_actual:mnist.train.labels[:10000], keep_prob:1.0})
    print("Final validation accuracy: {}".format(val_acc))
    print("Final training accuracy: {}".format(train_acc))
    return train_acc, val_acc

#nn(epochs = 20, regularizer = None)

#Good network with dropout
# Final validation accuracy: 0.9833999872207642
# Final training accuracy: 0.9966909289360046

# nn(layers = [1200,1200],
#         epochs = 20, 
#         dropout = True, 
#         keep_probability = 0.5,
#         regularizer=tf.contrib.layers.l2_regularizer(1e-3),
#         )
def main():
   for learning_rate in [1e-3, 1e-4]:
       for fc_layers in ["small", "big"]:
           if fc_layers == "big":
               layers = [800,800]
           else: 
               layers = [200,200]
           for conv_layers in [3, 4]:
               nn(layers = layers,
                   learning_rate = learning_rate,
                   epochs = 1, 
                   dropout = True, 
                   keep_probability = 0.5,
                   regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                   conv_layers = conv_layers,
                   exp_num = "learning_rate={}fc_layers={}conv_layers={}".format(learning_rate, fc_layers, conv_layers),
                   )
    # nn(layers = [400, 400],
    #     learning_rate = learning_rate,
    #     epochs = 1, 
    #     dropout = True, 
    #     keep_probability = 0.5,
    #     regularizer=tf.contrib.layers.l2_regularizer(1e-3),
    #     conv_layers = 4,
    #     exp_num = 2,
    #     )

if __name__ == '__main__':
    main()
