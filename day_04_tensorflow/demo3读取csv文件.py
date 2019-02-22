import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def csvread(filelist):
    """
    读取csv文件
    :param filelist: 文件路径列表
    :return: 返回读取的内容
    """

    # 1.构造文件队列
    file_queue = tf.train.string_input_producer(filelist)
    # 2.构造阅读器读取队列数据
    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)
    # print(value)

    # 对读取的数据进行解码
    # record_default:指定每一个样本的每一列的数据类型，指定默认值
    records = [["None"], ["None"]]
    example, label = tf.decode_csv(value, record_defaults=records)

    print(example, label)
    # 想要读取多个数据，进行批处理
    example_batch, label_batch = tf.train.batch([example, label], batch_size=9, num_threads=1, capacity=9)

    print(example_batch, label_batch)
    return example_batch, label_batch


if __name__ == '__main__':
    # 找到文件 构造列表
    file_name = os.listdir("./csvdata")

    filelist = [os.path.join('./csvdata', file) for file in file_name]
    example_batch, label_batch = csvread(filelist)
    # 开启会话运行结果
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启线程进行数据读取
        threads = tf.train.start_queue_runners(sess, coord)
        print(sess.run([example_batch, label_batch]))
        # 回收子线程
        coord.request_stop()
        coord.join(threads)
