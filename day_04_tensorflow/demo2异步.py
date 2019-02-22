import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 模拟异步线程存储数据，主线程读取数据

# 1.定义一个队列1000
Q = tf.FIFOQueue(1000, tf.float32)
# 2.定义要做的事情，循环值+1，放入队列当中
var = tf.Variable(0.0, tf.float32)

# 实现自增
data = tf.assign_add(var, tf.constant(1.0))

en_q = Q.enqueue(data)
# 3.定义队列管理器op，指定多少个子线程，子线程该干什么
qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)

# 初始化变量
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    # 开启子线程

    coord = tf.train.Coordinator()
    threads = qr.create_threads(sess, coord=coord, start=True)

    # 主线程读取数据
    for i in range(300):
        print(sess.run(Q.dequeue()))

    coord.request_stop()
    coord.join(threads)
