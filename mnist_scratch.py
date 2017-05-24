
import numpy as np
import tensorflow as tf
import pdb
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape = [None, 784])
y_actual = tf.placeholder(tf.float32, shape = [None, 10])

W = tf.Variable(tf.random_normal([784,10], stddev = 0.2))
# W = tf.zeros([784,10])
b = tf.Variable(tf.zeros([10]))

preactivation = tf.matmul(x, W) + b
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=preactivation, labels=y_actual)
mean_cross_entropy = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(mean_cross_entropy)
correct_prediction = tf.equal(tf.argmax(preactivation, 1), tf.argmax(y_actual, 1)) # finds label
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # reduce_mean requires float

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

iterations = 50
for _ in range(iterations):
  batch = mnist.train.next_batch(100)
  # pa = sess.run(preactivation, feed_dict={x: batch[0], y_actual: batch[1]})
  # pdb.set_trace()
  _, acc = sess.run([train_step, accuracy], feed_dict={x: batch[0], y_actual: batch[1]})
  print(acc)
