# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: CNN.py
   create time: Fri 18 Aug 2017 05:36:12 AM EDT
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
#An implement of LSTM
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist
import numpy as np

batch_size = 128

data = mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)

#input
GTX = tf.placeholder(tf.float32, [None, 28, 28]) #one line is a step
GTY = tf.placeholder(tf.float32, [None, 10])

def LSTM(X):
    X1 = tf.reshape(X, [-1, 28]) #batchsize*step input
    w1 = tf.Variable(tf.random_normal([28, 128]), dtype=tf.float32)
    b1 = tf.Variable(tf.constant(0.1, shape=[128]))
    w2 = tf.Variable(tf.random_normal([128, 10]), dtype=tf.float32)
    b2 = tf.Variable(tf.constant(0.1, shape=[10]))

    X2 = tf.matmul(X1, w1) + b1 #batchsize*step hiddennum
    X2 = tf.reshape(X2, [-1, 28, 128]) #batchsize step hiddennum

    lstmCell = tf.nn.rnn_cell.BasicLSTMCell(128, forget_bias=0.0, state_is_tuple=True)
    initState = lstmCell.zero_state(batch_size, tf.float32)

    X3, state = tf.nn.dynamic_rnn(lstmCell, X2, initial_state=initState, time_major=False) #batchsize step hiddennum -> time_major = false

    return tf.matmul(state[1], w2) + b2

if __name__ == "__main__":
    prob = LSTM(GTX)
    loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = GTY, logits = prob)))
    train = tf.train.AdamOptimizer(0.01).minimize(loss)

    correct_num = tf.equal(tf.arg_max(prob, 1), tf.arg_max(GTY, 1))
    acc = tf.reduce_mean(tf.cast(correct_num, dtype = tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000000):
            batch_X, batch_Y = data.train.next_batch(batch_size)

            batch_X = batch_X.reshape([batch_size, 28, 28])
            sess.run(train, feed_dict={GTX: batch_X, GTY: batch_Y})
            if i % 100 == 0:
                testacc = sess.run(acc, feed_dict={GTX: batch_X, GTY: batch_Y})
                print("step %d: %.3f" % (i, testacc))






