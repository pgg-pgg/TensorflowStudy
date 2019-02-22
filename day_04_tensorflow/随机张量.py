import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 正态分布的 4X4X4 三维矩阵，平均值 0， 标准差 1
input1 = tf.constant(3.0, name='input1')
input2 = tf.constant(4.0, name='input2')
sum1 = tf.add(input1, input2, name='sum1')

a = tf.Variable(tf.random_normal(shape=[2, 2], seed=1, mean=0.0, stddev=1.0, dtype=tf.float32))
b = tf.Variable(tf.random_normal(shape=[2, 2], seed=2, mean=0.0, stddev=1.0, dtype=tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sum1))
    file = tf.summary.FileWriter('./summary/test/', graph=sess.graph)
    print(sess.run(a))
    print(sess.run(b))
