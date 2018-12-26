import tensorflow as tf
import numpy as np
import os
import cv2
import re
from PIL import Image

class FeatureIO(object):
    def __init__(self):
        return

    def int64_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        value_tmp = []
        is_int = True
        for val in value:
            if not isinstance(val, int):
                is_int = False
                value_tmp.append(int(float(val)))
        if is_int is False:
            value = value_tmp
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def float_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        is_float = True
        value_tmp = []
        for val in value:
            if not isinstance(val, float):
                is_float = False
                value_tmp.append(float(val))
        if is_float is False:
            value = value_tmp
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def bytes_feature(self, value):
        if not isinstance(value, bytes):
            if not isinstance(value, list):
                value = value.encode('utf-8')
            else:
                value = [val.encode('utf-8') for val in value]
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class TextFeatureWriter(FeatureIO):
    def __init__(self):
        super(TextFeatureWriter, self).__init__()
        return

    def write_features(self, tfrecord_path, images, labels):
        assert len(images) == len(labels)
        if not os.path.exists(os.path.split(tfrecord_path)[0]):
            os.makedirs(os.path.split(tfrecord_path)[0])
        with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
            for index, image in enumerate(images):
                features = tf.train.Features(feature={
                    'labels' : self.int64_feature(labels[index]),
                    'images' : self.bytes_feature(image)
                })
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
        return

class TextFeatureReader(FeatureIO):
    def __init__(self):
        super(TextFeatureReader, self).__init__()
        return

    def read_features(self, tfrecords_dir, num_epochs, flag):
        assert os.path.exists(tfrecords_dir)

        if not isinstance(flag, str):
            raise ValueError('flag should be a str in [Train, Test, Validation]')
        if flag not in ['Train', 'Test', 'Validation']:
            raise ValueError('flag should be a str in [Train, Test, Validation]')

        if flag.lower() == 'train':
            re_patten = r'^train_feature_\d{0,15}_\d{0,15}\.tfrecords\Z'
        if flag.lower() == 'test':
            re_patten = r'^test_feature_\d{0,15}_\d{0,15}\.tfrecords\Z'
        if flag.lower() == 'validation':
            re_patten = r'^validation_feature_\d{0,15}_\d{0,15}\.tfrecords\Z'

        tfrecords_list = [os.path.join(tfrecords_dir, tmp) for tmp in os.listdir(tfrecords_dir) if re.match(re_patten, tmp)]
        filename_queue = tf.train.string_input_producer(tfrecords_list, num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'images' : tf.FixedLenFeature((), tf.string),
                                               'labels' : tf.VarLenFeature(tf.int64)
                                           })
        image = tf.decode_raw(features['images'], tf.uint8)
        images = tf.reshape(image, [32, 100, 3])
        labels = features['labels']
        labels = tf.cast(labels, tf.int32)
        # labels = tf.sparse_tensor_to_dense(labels)

        return images, labels


if __name__ == '__main__':
    Dataset_dir = "/home/oushu/lixingyu/DataSet/Synthetic_Chinese_String_Dataset"
    Test_dir = '/home/oushu/lixingyu/git_repo/attention_OCR/test'
    labels = []
    imagenames = []
    with open(os.path.join(Dataset_dir, 'data_test.txt')) as test_reader:
        for i, line in enumerate(test_reader.readlines()):
            line = line.strip().split(' ')
            if i < 32:
                labels.append(line[1:])
                imagenames.append(line[0])
            else:
                break
    images = [cv2.imread(os.path.join(Dataset_dir+'/images', name)) for name in imagenames]
    images = [cv2.resize(image, (100, 32)) for image in images]
    images = [bytes(list(np.reshape(tmp, [100*32*3]))) for tmp in images]
    FeatureWriter = TextFeatureWriter()
    FeatureWriter.write_features(os.path.join(Test_dir, 'test_feature_22240671_1034114836.tfrecords'), images, labels)
    FeatureReader = TextFeatureReader()
    test_images, test_labels = FeatureReader.read_features(os.path.join(Test_dir), 2, 'Test')
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        images, labels = sess.run([test_images, test_labels])
        image = Image.fromarray(images)
        image.show()
        print(labels)
        coord.request_stop()
        coord.join(threads=threads)
