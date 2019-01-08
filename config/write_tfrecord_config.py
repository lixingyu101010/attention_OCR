from easydict import EasyDict as edict

cfg = edict()

cfg.DATASET_DIR = '/home/oushu/tongli/ocr/ocr/TextGen/TextRecognitionDataGenerator/digitset1'#'/home/oushu/tongli/ocr/ocr/TextGen/TextRecognitionDataGenerator/newcorpus2'

cfg.IMAGE_NAME = 'images'

cfg.TRAIN_ANNO_NAME = 'data_train.txt'

cfg.TEST_ANNO_NAME = 'data_test.txt'

cfg.VALIDATION_SET = True

cfg.VALIDATION_SPLIT = 0.05

cfg.SHUFFLE = 'every_epoch'

cfg.NORMALIZATION = None

cfg.TRAIN_BATCH_SIZE = 32

cfg.SAVE_DIR = '/home/oushu/data/lxy/Dataset6_digest'

cfg.TEST_BATCH_SIZE = 32

cfg.VALIDATION_BATCH_SIZE = 32

cfg.IMAGE_WIDTH = 350

cfg.IMAGE_HIGHT = 32


if __name__ == '__main__':
    print(cfg.DATASET_DIR)

