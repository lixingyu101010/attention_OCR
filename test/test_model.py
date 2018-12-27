import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import sys
import os
import re
import cv2
from PIL import Image
import logging
sys.path.append(os.getcwd())
from config.global_config import CFG
import models.crnn_model as crnn_model

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stderr,
    format='%(levelname)s '
    '%(asctime)s: '
    '%(filename)s: '
    '%(lineno)d '
    '%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

class Decoder(object):
    def __init__(self, filename):
        self.id_2_word = []
        with open(filename, 'r') as reader:
            for i, line in enumerate(reader.readlines()):
                if i == 2:
                    print('==============',line[0])
                self.id_2_word.append(line[0])

    def decoder(self, ids):
        words = []
        for id in ids:
            print(id)
            words.append(self.id_2_word[int(id)])
        return ''.join(words)
with tf.device('/cpu:0'):
    # test_image = cv2.imread('/home/oushu/lixingyu/DataSet/image_chinese_dataset/images_chinese/img_0130106.jpg')
    test_image = cv2.imread('/home/oushu/lixingyu/DataSet/Synthetic_Chinese_String_Dataset/images/50850296_910274810.jpg')
    test_image = cv2.resize(test_image, (100, 32))
    img = Image.fromarray(test_image)
    img.show()

    test_data = tf.constant(test_image, tf.float32)
    # test_data = test_data / tf.constant(266, tf.float32)
    test_data = tf.expand_dims(test_data, 0)
    test_data = tf.cast(test_data, tf.float32)

    phase_tensor = tf.placeholder(dtype=tf.string, shape=None, name='phase')

    network = crnn_model.ShadowNet(phase=phase_tensor, is_train=False)

    _, ids, scores, tensor_dict = network.build_shadownet(test_data, None)

    sess = tf.Session()
    summary_writer = tf.summary.FileWriter('/home/oushu/lixingyu/git_repo/attention_OCR/tfboard/tb4')
    summary_writer.add_graph(sess.graph)

    saver = tf.train.Saver()

    with sess.as_default():
        # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # sess.run(init_op)
        saver.restore(sess, save_path='/home/oushu/lixingyu/git_repo/attention_OCR/test/checkpoint2/attention_network_2018-12-26-17-23-28.ckpt-505000')
        predict_ids = sess.run([ids], feed_dict={phase_tensor: 'test'})

        # word_decoder = Decoder('/home/oushu/lixingyu/DataSet/Synthetic_Chinese_String_Dataset/char_std_5990.txt')
        print(predict_ids[0][0])
        # words = word_decoder.decoder(predict_ids)

        # print(words)

# python test/test_model.py
# /home/oushu/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
#   from ._conv import register_converters as _register_converters
# [1, 32, 100, 3]
# INFO:tensorflow:Scale of 0 disables regularizer.
# INFO 2018-12-26 10:05:50: tf_logging.py: 115 Scale of 0 disables regularizer.
# display-im6.q16: unable to open X server `localhost:11.0' @ error/display.c/DisplayImageCommand/432.
# 2018-12-26 10:05:53.226781: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties:
# name: GeForce RTX 2080 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.635
# pciBusID: 0000:65:00.0
# totalMemory: 10.73GiB freeMemory: 1.55GiB
# 2018-12-26 10:05:53.226816: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
# 2018-12-26 10:05:53.926599: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2018-12-26 10:05:53.926642: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0
# 2018-12-26 10:05:53.926650: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N
# 2018-12-26 10:05:53.926841: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1286 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:65:00.0, compute capability: 7.5)
# INFO:tensorflow:Restoring parameters from /home/oushu/lixingyu/git_repo/attention_OCR/checkpoint/checkpoint1/attention_network_2018-12-25-18-30-00.ckpt-341000
# INFO 2018-12-26 10:05:54: tf_logging.py: 115 Restoring parameters from /home/oushu/lixingyu/git_repo/attention_OCR/checkpoint/checkpoint1/attention_network_2018-12-25-18-30-00.ckpt-341000
# [array([[311, 311, 162, 162,   1,  11,  11,  11, 473, 311, 311, 311, 311,
#         311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311]],
#       dtype=int32)]

# if __name__ == '__main__':
#     checkpoint_path = '/home/oushu/lixingyu/git_repo/attention_OCR/checkpoint/checkpoint1/attention_network_2018-12-25-18-30-00.ckpt-404000'
#     reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
#     var_to_shape_map = reader.get_variable_to_shape_map()
#     for key in var_to_shape_map:
#         print("tensor_name: ", key)
#         # print(reader.get_tensor(key))