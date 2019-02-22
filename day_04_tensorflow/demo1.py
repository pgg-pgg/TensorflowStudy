import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(5.0)
b = tf.constant(6.0)
print(a, b)
print(tf.get_default_graph, a.graph, b.graph)

sum1 = tf.add(a, b)

# 创建两个个浮点数占位符op
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

result = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(sum1))
    print(sess.run(result, feed_dict={input1: 7.0, input2: 9.0}))
