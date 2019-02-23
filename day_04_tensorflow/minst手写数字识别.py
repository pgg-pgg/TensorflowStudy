import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("is_train", "1", "是否在训练")


def full_connect():
    """
    全连接层
    :return:
    """
    # 获取真实数据
    mnist = input_data.read_data_sets("./mnist", one_hot=True)
    # 1。建立数据占位符 x[None,784] y_true[None,10]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.int32, [None, 10])

    # 2.建立一个全连接层的神经网络
    with tf.variable_scope("fc_model"):
        # 随机初始化weight和b
        weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name="weight")
        bias = tf.Variable(tf.constant(0.0, shape=[10]))

        # 预测None个样本的输出结果
        y_predict = tf.matmul(x, weight) + bias

    # 求出所有样本损失，求平均值
    with tf.variable_scope("soft_cross"):
        # 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 梯度下降求出损失
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 计算准确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        acc = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 收集变量
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", acc)

    tf.summary.histogram('weights', weight)
    tf.summary.histogram('biases', bias)

    # 初始化变量
    init_op = tf.global_variables_initializer()

    # 定义合并变量的op
    merged = tf.summary.merge_all()

    # 创建saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        # 建立event文件
        file_writer = tf.summary.FileWriter('./summary/test', graph=sess.graph)

        if FLAGS.is_train == "1":
            # 迭代步数去训练，更新参数预测
            for i in range(2000):
                # 取出真实的特征值与目标值
                mnist_x, mnist_y = mnist.train.next_batch(50)
                sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
                summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})
                file_writer.add_summary(summary, i)
                print("训练第%d步,准确率为：%s" % (i, sess.run(acc, feed_dict={x: mnist_x, y_true: mnist_y})))

            # 保存模型
            saver.save(sess, "./summary/mnistmodel/fc_model")

        else:
            saver.restore(sess, "./summary/mnistmodel/fc_model")
            # 开始测试
            for i in range(100):
                # 每次测试一张图片
                x_test, y_test = mnist.test.next_batch(1)
                print("第%d图片，目标是：%d,预测结果是：%d" %
                      (i,
                       tf.argmax(y_test, 1).eval(),
                       tf.argmax(sess.run(y_predict, feed_dict={x: x_test, y_true: y_test}),1).eval()
                       ))


if __name__ == '__main__':
    full_connect()
