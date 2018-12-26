import tensorflow as tf
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

string = 'attention_network_2018-12-25-11-49-34.ckpt-25300.meta'
matchObj = re.search(r'-([0-9]*)\.meta', string)
logging.info(matchObj.group(1))

# class Decoder(object):
#     def __init__(self, filename):
#         self.id_2_word = {}
#         with open(filename, 'r') as reader:
#             for i, line in enumerate(reader.readlines()):
#                 self.id_2_word[i] = line.strip().split()[0]
#
#     def decoder(self, ids):
#         words = []
#         for id in ids:
#             words.append(self.id_2_word[id])
#         return ''.join(words)
#
# test_image = cv2.imread('/home/oushu/lixingyu/DataSet/image_chinese_dataset/images_chinese/img_0130106.jpg')
# # test_image = cv2.imread('/home/oushu/lixingyu/DataSet/Synthetic_Chinese_String_Dataset/images/50850296_910274810.jpg')
# test_image = cv2.resize(test_image, (100, 32))
# img = Image.fromarray(test_image)
# img.show()
#
# test_data = tf.constant(test_image, tf.float32)
# test_data = test_data / tf.constant(266, tf.float32)
# test_data = tf.expand_dims(test_data, 0)
# print(test_data.shape.as_list())
#
# phase_tensor = tf.placeholder(dtype=tf.string, shape=None, name='phase')
#
# network = crnn_model.ShadowNet(phase=phase_tensor)
#
# _, ids, logit_prob, scores, tensor_dict = network.build_shadownet(test_data, None)
#
# sess = tf.Session()
#
# saver = tf.train.Saver()
#
# with sess.as_default():
#     saver.restore(sess, save_path='/home/oushu/lixingyu/git_repo/attention_OCR/checkpoint')
#
#     predict_ids = sess.run([ids], feed_dict={phase_tensor: 'test'})
#
#     print(predict_ids)

