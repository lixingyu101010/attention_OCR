import tensorflow as tf
import os.path as ops
import numpy as np
import copy
import cv2
from PIL import Image

class Dateset(object):
    def __init__(self):
        pass

    def shuffle_images_labels(self, imagenames, labels):
        imagenames = np.array(imagenames)
        labels = np.array(labels)

        assert imagenames.shape[0] == labels.shape[0]

        random_index = np.random.permutation(imagenames.shape[0])
        shuffled_labels = labels[random_index]
        shuffled_imagenames = imagenames[random_index]

        return shuffled_imagenames, shuffled_labels

    def normalize_images(self, images, normalization_type):
        if normalization_type == 'divide_255':
            for i, image in enumerate(images):
                images[i] = image / 255
        elif normalization_type == 'divide_256':
            for i, image in enumerate(images):
                images[i] = image / 256
        elif normalization_type is None:
            pass
        else:
            raise Exception('Unkown type of normalization')
        return images

    def normalization_image_by_channel(self, image):
        new_image = np.zeros(image.shape)
        for channel in range(3):
            mean = np.mean(image[:, :, channel])
            std = np.std(image[:, :, channel])
            new_image[:, :, channel] = (image[:, :, channel] - mean) / std
        return new_image

    def normalization_all_images_by_channels(self, initial_images):
        new_images = np.zeros(initial_images.shape)
        for i in range(initial_images.shape[0]):
            new_images[i] = self.normalization_image_by_channel(initial_images[i])
        return new_images

class TextDataset(Dateset):
    def __init__(self, imagenames, labels, shuffle=None, normalization=None):
        super(TextDataset, self).__init__()

        self.normalization = normalization
        if self.normalization not in [None, 'divide_255', 'divide_256']:
            raise ValueError('normalization should be in [None, divide_255, divide_256]')
        self.labels = labels
        self.imagenames = imagenames
        self.num_example = imagenames.shape[0]
        self.epoch_labels = copy.deepcopy(self.labels)
        self.epoch_imagenames = copy.deepcopy(self.imagenames)

        self.shuffle = shuffle
        if self.shuffle not in [None, 'once_prior_train', 'every_epoch']:
            raise ValueError('shuffle parameter wrong')
        if self.shuffle == 'every_epoch' or 'once_prior_train':
            self.epoch_imagenames, self.epoch_labels = self.shuffle_images_labels(self.epoch_imagenames, self.epoch_labels)
            # print(self.epoch_imagenames[0])
            # print(self.epoch_labels[0])


        self.batch_counter = 0

    def start_new_epoch(self):
        self.batch_counter = 0
        if self.shuffle == 'every_epoch':
            self.epoch_imagenames, self.epoch_labels = self.shuffle_images_labels(self.epoch_imagenames, self.epoch_labels)
        else:
            pass
        return

    def next_batch(self, batch_size):
        start = self.batch_counter * batch_size
        end = (self.batch_counter + 1) * batch_size
        self.batch_counter += 1

        imagenames_slice = self.epoch_imagenames[start:end]
        labels_slice = self.epoch_labels[start:end]
        # print(imagenames_slice[0])
        # print(labels_slice[0])
        images_slice = [cv2.imread(tmp, cv2.IMREAD_UNCHANGED) for tmp in imagenames_slice]
        images_slice = self.normalize_images(images_slice, self.normalization)

        if len(images_slice) != batch_size and self.batch_counter > 1:
            self.start_new_epoch()
            return images_slice, labels_slice
        else:
            return images_slice, labels_slice

# class TextDataProvider(object):
#     def __init__(self, dataset_dir, annotation_name, validation_set=None, validation_split=None, shuffle=None, normalization=None):
#         self.dataset_dir = dataset_dir
#         self.validation_dir = validation_split
#         self.shuffle = shuffle
#         self.normalization = normalization
#         self.train_dataset_dir = ops.join(self.dataset_dir, 'Train')
#         self.test_dataset_dir = ops.join(self.dataset_dir, 'Test')
#
#         assert ops.exists(dataset_dir)
#         assert ops.exists(self.train_dataset_dir)
#         assert ops.exists(self.test_dataset_dir)
#
#         test_anno_path = ops.join(self.test_dataset_dir, annotation_name)
#         assert ops.exists(test_anno_path)
#
#         with open(test_anno_path, 'r') as anno_file:
#             info = np.array([tmp.strip().split() for tmp in anno_file.readlines()])
#             test_labels = np.array([tmp for tmp in info[:, 1]])
#             test_imagesnames = np.array([ops.join(self.test_dataset_dir, tmp) for tmp in info[:, 0]])
#             self.test = TextDataset(labels=test_labels, imagenames=test_imagenames, shuffle=shuffle, normalization=normalization)
#         anno_file.close()
#
#         train_anno_path = ops.join(self.train_dataset_dir, annotation_name)
#         assert ops.exists(train_anno_path)
#
#         with open(train_anno_path, 'r') as anno_file:
#             info = np.array([tmp.strip().split() for tmp in anno_file.readlines()])
#             train_labels = np.array([tmp for tmp in info[:, 1]])
#             train_imagenames = np.array([ops.join(self.train_dataset_dir, tmp) for tmp in info[:, 0]])
#
#             if validation_set is not None and validation_split is not None:
#                 split_idx = int(train_imagenames.shape[0] * (1 - validation_split))
#                 self.train = TextDataset(labels=train_labels[:split_idx], shuffle=shuffle,
#                                          normalization=normalization, imagenames=train_imagenames[:split_idx])
#
#                 self.validation = TextDataset(labels=train_labels[split_idx:], shuffle=shuffle,
#                                               normalization=normalization, imagenames=train_imagenames[split_idx:])
#             else:
#                 self.train = TextDataset(labels=train_labels, shuffle=shuffle, normalization=normalization, imagenames=train_imagenames)
#
#             if validation_set and not validation_split:
#                 self.validation = self.test
#         anno_file.close()
#         return

class TextDataProvider(object):
    def __init__(self, dataset_dir, images_name, train_anno_name, test_anno_name,  validation_set=None, validation_split=None, shuffle=None, normalization=None):

        test_anno_dir = ops.join(dataset_dir, test_anno_name)
        train_anno_dir = ops.join(dataset_dir, train_anno_name)
        images_dir = ops.join(dataset_dir, images_name)

        assert ops.exists(test_anno_dir)
        assert ops.exists(train_anno_dir)
        assert ops.exists(images_dir)

        with open(test_anno_dir, 'r') as anno_file:
            info = np.array([tmp.strip().split(' ') for tmp in anno_file.readlines()])
            test_labels = np.array([tmp[1:] for tmp in info])
            test_imagenames = np.array([ops.join(images_dir, tmp[0]) for tmp in info])
            self.test = TextDataset(labels=test_labels, shuffle=shuffle, normalization=normalization, imagenames=test_imagenames)

        with open(train_anno_dir, 'r') as anno_file:
            info = np.array([tmp.strip().split(' ') for tmp in anno_file.readlines()])
            train_labels = np.array([tmp[1:] for tmp in info])
            train_imagenames = np.array([ops.join(images_dir, tmp[0]) for tmp in info])

            if validation_set is not None and validation_split is not None:
                split_idx = int(train_imagenames.shape[0] * (1 - validation_split))
                self.train = TextDataset(labels=train_labels[:split_idx], shuffle=shuffle,
                                         normalization=normalization, imagenames=train_imagenames[:split_idx])
                self.validation = TextDataset(labels=train_labels[split_idx:], shuffle=shuffle,
                                              normalization=normalization, imagenames=train_imagenames[split_idx:])
            else:
                self.train = TextDataset(labels=train_labels, shuffle=shuffle, normalization=normalization, imagenames=train_imagenames)

            if validation_set and validation_split is None:
                self.validation = self.test
        anno_file.close()
        return






if __name__ == '__main__':
    dataset_dir = '/home/oushu/lixingyu/DataSet/Synthetic_Chinese_String_Dataset'
    images_name = 'images'
    train_anno_name = 'data_train.txt'
    test_anno_name = 'data_test.txt'
    data_provider = TextDataProvider(dataset_dir, images_name, train_anno_name, test_anno_name, True, None, 'every_epoch', 'divide_256')
    images, labels = data_provider.test.next_batch(2)
    print(images.shape)
    print(labels)


    # Test_dir = '/home/oushu/lixingyu/git_repo/attention_OCR/test'
    # FeatureIO = data_utils.TextFeatureReader()
    # images, labels = FeatureIO.read_features(Test_dir, 1, 'Test')
    # test_images, test_labels = tf.train.shuffle_batch(
    #     tensors=[images, labels], batch_size=1,
    #     capacity=1000 + 2 * 32, min_after_dequeue=100, num_threads=2
    # )
    # with tf.Session() as sess:
    #     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #     sess.run(init_op)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     output_images, output_labels = sess.run([test_images, test_labels])
    #     print(output_images)
    #     print(output_labels)
    #     coord.request_stop()
    #     coord.join(threads)
