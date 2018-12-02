#-*- coding:utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

# data = np.loadtxt('cocktail.csv', delimiter=',', dtype=np.float32)
# x_data = data[:, 0:5]
# y_data = data[:, 6:]
#
# X = tf.placeholder(tf.float32, shape=[None,5])
# Y = tf.placeholder(tf.float32, shape=[None,60])
#
# print(x_data)
# print(y_data)
# nb_classes = 60
#
# W = tf.Variable(tf.random_normal([5,nb_classes]),name='weight')
# b = tf.Variable(tf.random_normal([nb_classes]),name='bias')
#
# hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(20001):
#         sess.run(optimizer, feed_dict={X:x_data,Y:y_data})
#         if step % 400 == 0:
#             print(step, sess.run(cost, feed_dict={X:x_data,Y:y_data}))
#     a = sess.run(hypothesis, feed_dict={X: [[1, 0, 1 ,1, 1]]})
#     print(a, sess.run(tf.argmax(a,1)))


user_data = np.loadtxt('user.csv', delimiter=',', dtype=np.float32, encoding='UTF8')
user_x_data = user_data[:, 0:5]
user_y_data = user_data[:, 6:]

print(user_x_data)
print(user_y_data)

X = tf.placeholder(tf.float32, shape=[None,5])
Y = tf.placeholder(tf.float32, shape=[None,60])

nb_classes = 60

W = tf.Variable(tf.random_normal([5,nb_classes]),name='weight')
b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        sess.run(optimizer, feed_dict={X:user_x_data,Y:user_y_data})
        if step % 400 == 0:
            print(step, sess.run(cost, feed_dict={X:user_x_data,Y:user_y_data}))
    a = sess.run(hypothesis, feed_dict={X: [[1, 16, 17 ,25, 43]]})
    print(a, sess.run(tf.argmax(a,1)))