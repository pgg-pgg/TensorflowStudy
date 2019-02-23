import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义数据文件路径的命令行参数
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("cifar_fir", "./binarydata/cifar-10-batches-bin/", "文件的目录")
tf.app.flags.DEFINE_string("cifar_tfrecords", "./binarydata/cifar_tfrecords.txt", "tfrecords存储的目录")


class CifarReader(object):
    """
    读取Cifar文件的类
    """

    def __init__(self, filelist):
        self.filelist = filelist
        # 定义读取文件的一些属性
        self.height = 32
        self.width = 32
        self.channel = 3

        # 二进制每张图片的每张大小
        self.label_bytes = 1
        self.image_bytes = self.height * self.width * self.channel
        self.bytes = self.label_bytes + self.image_bytes

    def read_binary(self):
        # 1。构造文件队列'
        file_queue = tf.train.string_input_producer(self.filelist)

        # 2。构造文件读取器，读取文件
        reader = tf.FixedLengthRecordReader(self.bytes)
        key, value = reader.read(file_queue)

        # 3。解码
        label_image = tf.decode_raw(value, tf.uint8)

        # 4.分割出图片的label和image
        label = tf.cast(tf.slice(label_image, [0], [self.label_bytes]), tf.uint32)

        image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])

        # 5.对图片的形状进行定义
        image_reshape = tf.reshape(image, [self.width, self.height, self.channel])

        # 6.批处理数据
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)

        return image_batch, label_batch

    def write_to_records(self, image_batch, label_batch):
        """
        将图片特征值与目标值存储在tfRecords
        :param image_batch:10张图片的特征值
        :param label_batch:10张图片的目标值
        :return:
        """

        # 构造tfrecords存储器
        writer = tf.python_io.TFRecordWriter(FLAGS.cifar_tfrecords)

        # 循环将每张图片写入
        for i in range(10):
            image = image_batch[i].eval().tostring()
            label = label_batch.eval()[i][0]

            # 构造example协议
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))

            # 写入单独的样本
            writer.write(example.SerializeToString())

        writer.close()
        return None

    def read_from_tfrecords(self):
        # 1.构造文件队列
        file_queue = tf.train.string_input_producer([FLAGS.cifar_tfrecords])

        # 2.构造文件阅读器
        reader = tf.TFRecordReader()
        key, value = reader.read(file_queue)

        # 3.解析example
        features = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        })
        print(features["image"], features["label"])

        # tf.reshape转换
        # 4.解码
        image = tf.decode_raw(features['image'], tf.uint8)

        # 对image的形状进行固定，方便批处理
        image_reshape = tf.reshape(image, [self.width, self.height, self.channel])
        label = features['label']
        print(image_reshape, label)

        # 进行批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label], 10, 1, 10)

        return image_batch, label_batch


if __name__ == '__main__':
    file_name = os.listdir(FLAGS.cifar_fir)

    filelist = [os.path.join(FLAGS.cifar_fir, file) for file in file_name if file[-3:] == 'bin']

    cifarReader = CifarReader(filelist)
    # image_batch, label_batch = cifarReader.read_binary()

    image_batch, label_batch = cifarReader.read_from_tfrecords()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 打印读取出的文件
        print(sess.run([image_batch, label_batch]))
        # print("开始存储")
        # cifarReader.write_to_records(image_batch, label_batch)
        # print("结束存储")
        coord.request_stop()
        coord.join(threads)
