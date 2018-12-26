from easydict import EasyDict as edict

cfg = edict()

cfg.DATASET_DIR = '/home/oushu/lixingyu/DataSet/Synthetic_Chinese_String_Dataset'

cfg.IMAGE_NAME = 'images'

cfg.TRAIN_ANNO_NAME = 'data_train.txt'

cfg.TEST_ANNO_NAME = 'data_test.txt'

cfg.VALIDATION_SET = True

cfg.VALIDATION_SPLIT = 0.05

cfg.SHUFFLE = 'every_epoch'

cfg.NORMALIZATION = None

cfg.TRAIN_BATCH_SIZE = 32

cfg.SAVE_DIR = '/home/oushu/lixingyu/git_repo/attention_OCR/Dataset'

cfg.TEST_BATCH_SIZE = 32

cfg.VALIDATION_BATCH_SIZE = 32


if __name__ == '__main__':
    print(cfg.DATASET_DIR)

