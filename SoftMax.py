#-*- coding:utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

cocktail_data = np.loadtxt('cocktail.csv', delimiter=',', dtype=np.int32)
cocktail_x_data = cocktail_data[:, 0:5]
cocktail_y_data = cocktail_data[:, 6:]

cocktail_X = tf.placeholder(tf.float32, shape=[None,5])
cocktail_Y = tf.placeholder(tf.float32, shape=[None,60])

nb_classes = 60

W = tf.Variable(tf.random_normal([5,nb_classes]),name='weight')
b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

cocktail_hypothesis = tf.nn.softmax(tf.matmul(cocktail_X,W)+b)
cocktail_cost = tf.reduce_mean(-tf.reduce_sum(cocktail_Y*tf.log(cocktail_hypothesis),axis=1))
cocktail_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cocktail_cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        sess.run(cocktail_optimizer, feed_dict={cocktail_X:cocktail_x_data,cocktail_Y:cocktail_y_data})
        if step % 400 == 0:
            print(step, sess.run(cocktail_cost, feed_dict={cocktail_X:cocktail_x_data,cocktail_Y:cocktail_y_data}))
    recommend1 = sess.run(cocktail_hypothesis, feed_dict={cocktail_X: [[2, 3, 0 ,0, 8]]})
    print("cocktail recommend : ",sess.run(tf.argmax(recommend1,1)+1))


user_data = np.loadtxt('user2.csv', delimiter=',', dtype=np.float32)
user_x_data = user_data[:, 0:5]
user_y_data = user_data[:, 6:]

user_X = tf.placeholder(tf.float32, shape=[None,5])
user_Y = tf.placeholder(tf.float32, shape=[None,60])

nb_classes = 60

user_W = tf.Variable(tf.random_normal([5,nb_classes]),name='weight')
user_b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

user_hypothesis = tf.nn.softmax(tf.matmul(user_X,W)+b)
user_cost = tf.reduce_mean(-tf.reduce_sum(user_Y*tf.log(user_hypothesis),axis=1))
user_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(user_cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(50001):
        sess.run(user_optimizer, feed_dict={user_X:user_x_data,user_Y:user_y_data})
        if step % 400 == 0:
            print(step, sess.run(user_cost, feed_dict={user_X:user_x_data,user_Y:user_y_data}))
    recommend2 = sess.run(user_hypothesis, feed_dict={user_X: [[0.3, 0.4, 2 ,2.5, 3]]})
    print("user recommend : ", sess.run(tf.argmax(recommend2,1)+1))