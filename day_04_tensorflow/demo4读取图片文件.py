import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'


def picread(filelist):
    """
    读取人脸图片
    :param filelist:文件路径加名字的列表
    :return: 每张图片的张量
    """
    # 构造文件队列
    file_queue = tf.train.string_input_producer(filelist)

    # 默认读取一张图片
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)

    print(key, value)

    # 对读取的图片数据进行解码
    image = tf.image.decode_jpeg(value)
    print(image)

    # 处理图片的大小
    image_resize = tf.image.resize_images(image, [200, 200])
    print(image_resize)
    # 把样本的shape固定
    image_resize.set_shape([200, 200, 3])
    # 进行批处理
    image_batch = tf.train.batch([image_resize], batch_size=20, num_threads=1, capacity=20)

    return image_batch


if __name__ == '__main__':
    # 找到文件 构造列表
    file_name = os.listdir("./picdata")

    filelist = [os.path.join('./picdata', file) for file in file_name]
    picread(filelist)
    image_batch = picread(filelist)
    # 开启会话运行结果
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启线程进行数据读取
        threads = tf.train.start_queue_runners(sess, coord)
        print(sess.run([image_batch]))
        # 回收子线程
        coord.request_stop()
        coord.join(threads)
