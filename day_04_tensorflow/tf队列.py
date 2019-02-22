import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建一个队列
Q = tf.FIFOQueue(3, tf.float32)

# 数据进队列
init = Q.enqueue_many(([0.1, 0.2, 0.3],))
# 定义操作,op，出队列，+1，进队列,注意返回的都是op
out = Q.dequeue()
data = out+1
en_q = Q.enqueue(data)


with tf.Session() as sess:
    sess.run(init)

    # 执行两次入队加1
    for i in range(2):
        sess.run(en_q)

    # 循环取队列
    for i in range(3):
        print(sess.run(Q.dequeue()))
