import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


def model():
    """
    自定义卷积神经网络
    :return:
    """
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.float32, [None, 10])

    # 卷积层1 卷积：5*5*1，32个，strides = 1，激活，tf.nn.relu 池化
    with tf.variable_scope("conv1"):
        # 初始化权重
        w_conv1 = init_weight([5, 5, 1, 32])
        b_conv1 = init_bias([32])
        # 对x进行形状的改变【None,784】=> [None,28,28,1]
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])

        # [None,28,28,1]=>[None,28,28,32]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)

        # 池化 2*2，strides2 [None,28,28,32]=>[None,14,14,32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 卷积层2 卷积：5*5*32，64个，strides = 1，激活，tf.nn.relu 池化
    with tf.variable_scope("conv2"):
        # 初始化权重
        w_conv2 = init_weight([5, 5, 32, 64])
        b_conv2 = init_bias([64])
        # [None,14,14,32]=>[None,14,14,64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)

        # 池化 2*2，strides2 [None,14,14,64]=>[None,7,7,64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 全连接层【None,7,7,64]=>[None,7*7*64]*[7*7*64,10]+[10] = [None,10]
    with tf.variable_scope("fc"):
        # 随机初始化权重和偏置
        w_fc = init_weight([7 * 7 * 64, 10])
        b_fc = init_bias([10])

        # 修改形状【None,7,7,64]=>None,7*7*64
        x_fc_reshape = tf.reshape(x_pool2, [-1, 7 * 7 * 64])
        # 进行矩阵运算的出每个样本的10个结果
        y_predict = tf.matmul(x_fc_reshape, w_fc) + b_fc

    return x, y_true, y_predict


def conv_fc():
    # 获取真实数据
    mnist = input_data.read_data_sets("./mnist", one_hot=True)

    # 定义模型，得出输出
    x, y_true, y_predict = model()

    # 求出所有样本损失，求平均值
    with tf.variable_scope("soft_cross"):
        # 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 梯度下降求出损失
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # 计算准确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        acc = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    init_op = tf.global_variables_initializer()
    # 开启会话运行
    with tf.Session() as sess:
        sess.run(init_op)

        # 循环去训练
        for i in range(2000):
            # 取出真实的特征值与目标值
            mnist_x, mnist_y = mnist.train.next_batch(50)
            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
            print("训练第%d步,准确率为：%s" % (i, sess.run(acc, feed_dict={x: mnist_x, y_true: mnist_y})))


def init_weight(shape):
    """
    初始化权重
    :return:
    """
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w


def init_bias(shape):
    bias = tf.constant(0.0, shape=shape)
    return bias


if __name__ == '__main__':
    conv_fc()
