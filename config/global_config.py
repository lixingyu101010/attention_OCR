from easydict import EasyDict as edict

CFG = edict()

CFG.CLASSES_NUMS = 6113
CFG.DATASET_DIR = "/home/oushu/data/lxy/Dataset6_digest"#'/home/oushu/lixingyu/git_repo/attention_OCR/Dataset/Dataset1'
CFG.MAX_SEQ_LEN = 22
CFG.NUM_LSTM_UNITS = 256
CFG.LSTM_STATE_CLIP_VAL = 10.0
CFG.BATCH_SIZE = 32
CFG.LEARNING_RATE = 0.01
CFG.LR_DECAY_STEPS = 100000
CFG.LR_DECAY_RATE = 0.9
CFG.TRAIN_STEPS = 1500000
CFG.DISPLAY_STEP = 10
CFG.SAVE_STEP = 1000
CFG.TB_SAVE_DIR = '/home/oushu/lixingyu/git_repo/attention_OCR/tfboard/tb1'
CFG.CK_SAVE_DIR = '/home/oushu/lixingyu/git_repo/attention_OCR/checkpoint/checkpoint1'
CFG.PRE_WEIGHTS = '/home/oushu/lixingyu/git_repo/attention_OCR/checkpoint/checkpoint1/attention_network_2019-01-05-15-38-06.ckpt-0'