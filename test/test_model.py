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

# class Decoder(object):
#     def __init__(self, filename):
#         self.id_2_word = []
#         with open(filename, 'r') as reader:
#             for i, line in enumerate(reader.readlines()):
#                 if i == 2:
#                     print('==============',line[0])
#                 self.id_2_word.append(line[0])
#
#     def decoder(self, ids):
#         words = []
#         for id in ids:
#             print(id)
#             words.append(self.id_2_word[int(id)])
#         return ''.join(words)

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
    # summary_writer = tf.summary.FileWriter('/home/oushu/lixingyu/git_repo/attention_OCR/tfboard/tb4')
    # summary_writer.add_graph(sess.graph)

    saver = tf.train.Saver()

    with sess.as_default():
        # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # sess.run(init_op)
        saver.restore(sess, save_path='/home/oushu/lixingyu/git_repo/attention_OCR/test/checkpoint2/attention_network_2018-12-26-17-23-28.ckpt-505000')
        predict_ids = sess.run([ids], feed_dict={phase_tensor: 'test'})

        # word_decoder = Decoder('/home/oushu/lixingyu/DataSet/Synthetic_Chinese_String_Dataset/char_std_5990.txt')
        print(predict_ids[0][0])
