import tensorflow as tf
import tqdm
import sys
import math
import os.path as ops
import numpy as np
import cv2
from PIL import Image

sys.path.append('/home/oushu/lixingyu/git_repo/attention_OCR')
import data.create_dataset as data_provider
from data import data_utils
# import data.data_utils as data_utils
from config.write_tfrecord_config import cfg

def write_tfrecord():
    print('Initialize the dataset provider .....')
    provider = data_provider.TextDataProvider(cfg.DATASET_DIR, cfg.IMAGE_NAME, cfg.TRAIN_ANNO_NAME, cfg.TEST_ANNO_NAME,
                                              cfg.VALIDATION_SET, cfg.VALIDATION_SPLIT, cfg.SHUFFLE, cfg.NORMALIZATION)
    print('Dataset provider initialize complete')
    Feature_io = data_utils.TextFeatureWriter()

    print('Start writing training tfrecords')
    train_images_num_example = provider.train.num_example
    epoch_nums = int(math.ceil(train_images_num_example / cfg.TRAIN_BATCH_SIZE))
    print('epoch_num:', epoch_nums, 'train_images_nums:', train_images_num_example, 'batch_size:', cfg.TRAIN_BATCH_SIZE)
    for loop in tqdm.tqdm(range(epoch_nums)):
        train_images, train_labels = provider.train.next_batch(batch_size=cfg.TRAIN_BATCH_SIZE)
        train_images = [cv2.resize(tmp, (100, 32)) for tmp in train_images]
        train_images = [bytes(list(np.reshape(tmp, [100*32*3]))) for tmp in train_images]
        train_labels = train_labels.tolist()
        if loop*cfg.TRAIN_BATCH_SIZE+cfg.TRAIN_BATCH_SIZE > train_images_num_example:
            train_tfrecord_path = ops.join(cfg.SAVE_DIR, 'train_feature_{:d}_{:d}.tfrecords'.format(
                loop*cfg.TRAIN_BATCH_SIZE, train_images_num_example
            ))
        else:
            train_tfrecord_path = ops.join(cfg.SAVE_DIR, 'train_feature_{:d}_{:d}.tfrecords'.format(
                loop * cfg.TRAIN_BATCH_SIZE, loop * cfg.TRAIN_BATCH_SIZE + cfg.TRAIN_BATCH_SIZE
            ))

        Feature_io.write_features(tfrecord_path=train_tfrecord_path, images=train_images, labels=train_labels)
    print('Start writing testing tfrecords')

    test_image_num_example = provider.test.num_example
    epoch_nums = int(math.ceil(test_image_num_example / cfg.TEST_BATCH_SIZE))
    print('epoch_num:', epoch_nums, 'test_images_num:', test_image_num_example, 'batch_size:', cfg.TEST_BATCH_SIZE)
    for loop in tqdm.tqdm(range(epoch_nums)):
        test_images, test_labels = data_provider.test.next_batch(batch_size=cfg.TEST_BATCH_SIZE)
        test_images = [cv2.resize(tmp, (100, 32)) for tmp in test_images]
        test_images = [bytes(list(np.shape(tmp, [32*100*3]))) for tmp in test_images]
        test_labels = test_labels.tolist()
        if loop*cfg.TEST_BATCH_SIZE+cfg.TEST_BATCH_SIZE > test_image_num_example:
            test_tfrecord_path = ops.join(cfg.SAVE_DIR, 'test_feature_{:d}_{:d}.tfrecords'.format(
                loop*cfg.TEST_BATCH_SIZE, test_image_num_example
            ))
        else:
            test_tfrecord_path = ops.join(cfg.SAVE_DIR, 'test_feature_{:d}_{:d}.tfrecords'.format(
                loop * cfg.TEST_BATCH_SIZE, loop * cfg.TEST_BATCH_SIZE + cfg.TEST_BATCH_SIZE
            ))
        Feature_io.write_features(tfrecord_path=test_tfrecord_path, images=test_images, labels=test_labels)

    print('Start writing validation tfrecords')

    validation_image_num_example = provider.validation.num_example
    epoch_nums = int(math.ceil(validation_image_num_example / cfg.VALIDATION_BATCH_SIZE))
    print('epoch_num:', epoch_nums, 'test_images_num:', validation_image_num_example, 'batch_size:', cfg.VALIDATION_BATCH_SIZE)
    for loop in tqdm.tqdm(range(epoch_nums)):
        validation_images, validation_labels = data_provider.validation.next_batch(cfg.VALIDATION_BATCH_SIZE)
        validation_images = [cv2.resize(tmp, (100,32)) for tmp in validation_images]
        validation_images = [bytes(list(np.reshape(tmp, [32*100*3]))) for tmp in validation_images]
        validation_labels = validation_labels.tolist()
        if loop*cfg.VALIDATION_BATCH_SIZE+cfg.VALIDATION_BATCH_SIZE > validation_image_num_example:
            validation_tfrecord_path = ops.join(cfg.SAVE_DIR, 'validation_feature_{:d}_{:d}.tfrecords'.format(
                loop*cfg.VALIDATION_BATCH_SIZE, validation_image_num_example
            ))
        else:
            validation_tfrecord_path = ops.join(cfg.SAVE_DIR, 'validation_feature_{:d}_{:d}.tfrecords'.format(
                loop * cfg.VALIDATION_BATCH_SIZE, loop * cfg.VALIDATION_BATCH_SIZE + cfg.VALIDATION_BATCH_SIZE
            ))

        Feature_io.wirte_features(tfrecord_path=validation_tfrecord_path, images=validation_images, labels=validation_labels)
    return

def test_reader_tfrecord():
    FeatureIO = data_utils.TextFeatureReader()
    images, labels = FeatureIO.read_features(cfg.SAVE_DIR, 1, 'Train')
    test_images, test_labels = tf.train.shuffle_batch(
        tensors=[images, labels], batch_size=2,
        capacity=1000 + 2 * 32, min_after_dequeue=100, num_threads=2
    )
    test_labels = tf.sparse_tensor_to_dense(test_labels)
    ld = []
    with open(ops.join(cfg.DATASET_DIR, 'char_std_5990.txt'), 'r') as fp:
        for line in fp.readlines():
            char = line.strip()
            ld.append(char)
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        output_images, output_labels = sess.run([test_images, test_labels])
        merge = None
        for i, image in enumerate(output_images):
            if i == 0:
                merge = np.concatenate([image, output_images[i+1]], axis=0)
            else:
                if i+1 > len(output_images) - 1:
                    break
                else:
                    merge = np.concatenate([merge, output_images[i+1]], axis=0)
        test_image = Image.fromarray(merge)
        test_image.show()
        output_labels = output_labels.tolist()
        for n,label in enumerate(output_labels):
            for i, x in enumerate(label):
                label[i] = ld[int(x)]
            output_labels[n] = ''.join(label)
        print(output_labels)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    if not ops.exists(cfg.DATASET_DIR):
        ops.mkdirs(cfg.DATASET_DIR)
    if not ops.exists(cfg.SAVE_DIR):
        ops.mkdirs(cfg.SAVE_DIR)
    write_tfrecord()
    # test_reader_tfrecord()