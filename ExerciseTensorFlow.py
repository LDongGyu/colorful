#-*- coding:utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

sess = tf.Session()
hello = tf.constant('Hello, TensorFlow')
print(sess.run(hello).decode(encoding='utf-8'))
