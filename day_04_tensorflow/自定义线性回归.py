import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def regression():
    """
    自定义实现线性回归
    :return:
    """
    with tf.variable_scope('data'):
        # 准备数据
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name='x_data')
        # 矩阵相乘必须是二维的
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope('model'):
        # 建立线性回归模型，随机给出w和b，进行优化
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name='weights')
        bias = tf.Variable(0.0, name='b')
        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope('loss'):
        # 建立损失函数
        loss = tf.reduce_mean(tf.square(y_true - y_predict), name='losses')
    with tf.variable_scope('optimizer'):
        # 梯度下降优化损失
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 收集tensor
    tf.summary.scalar("losses", loss)
    tf.summary.histogram('weights', weight)

    # 合并变量写入事件文件
    merge = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    # 定义一个保存当前的sess的saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        print("随机初始化的权重为：%f,偏置:%f" % (weight.eval(), bias.eval()))
        # 建立事件文件
        file = tf.summary.FileWriter('./summary/test/', graph=sess.graph)
        # 加载模型，从上次训练的结果之后进行训练
        if os.path.exists('./summary/ckpt/checkpoint'):
            saver.restore(sess, './summary/ckpt/model')

        for i in range(500):
            sess.run(train_op)
            summary = sess.run(merge)
            file.add_summary(summary, i)
            print("第%d次训练的权重为：%f,偏置:%f" % (i, weight.eval(), bias.eval()))

        saver.save(sess, './summary/ckpt/model')
    return None


if __name__ == '__main__':
    regression()
