import tensorflow as tf
import os
import sys
import time
import logging
import re
from PIL import Image
sys.path.append(os.getcwd())
from data import data_utils
from config.global_config import CFG
import models.crnn_model as crnn_model

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stderr,
    format='%(levelname)s:'
    '%(asctime)s '
    '%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def train_atteion_network(dataset_dir, weight_path=None):
    FeatureIO = data_utils.TextFeatureReader()
    images, labels = FeatureIO.read_features(dataset_dir, 20, 'Train')
    train_images, train_labels = tf.train.shuffle_batch(
        tensors=[images, labels], batch_size=CFG.BATCH_SIZE,
        capacity=1000 + 2 * 32, min_after_dequeue=100, num_threads=2
    )

    ground_labels = tf.sparse_to_dense(train_labels.indices, [CFG.BATCH_SIZE, CFG.MAX_SEQ_LEN], train_labels.values)

    phase_tensor = tf.placeholder(dtype=tf.string, shape=None, name='phase')

    inputdata = tf.cast(x=train_images, dtype=tf.float32)

    # inputdata = inputdata / tf.constant(266, dtype=tf.float32)

    network = crnn_model.ShadowNet(phase=phase_tensor)

    loss, ids, scores, tensor_dict = network.build_shadownet(inputdata, train_labels)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    start_learning_rate = CFG.LEARNING_RATE

    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                               CFG.LR_DECAY_STEPS, CFG.LR_DECAY_RATE, staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)

    tfboard_save_path = '/home/oushu/lixingyu/git_repo/attention_OCR/tfboard/tb3'

    train_loss_scalar = tf.summary.scalar(name='train_loss', tensor=loss)
    accuracy = tf.placeholder(tf.float32, shape=None, name='train_accuracy')
    train_accuracy_scalar = tf.summary.scalar(name='train_accuracy', tensor=accuracy)

    train_summary_op_merge = tf.summary.merge(inputs=[train_loss_scalar, train_accuracy_scalar])

    # restore_variable_list = [tmp.name for tmp in tf.trainable_variables()]

    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
    model_save_dir = '/home/oushu/lixingyu/git_repo/attention_OCR/checkpoint/checkpoint1'
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'attention_network_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(model_save_dir, model_name)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tfboard_save_path)
    summary_writer.add_graph(sess.graph)

    start_step = 0

    with sess.as_default():
        if weight_path is None:
            logging.info('Train from initial')
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
        else:
            logging.info('Train from checkpoint')
            sess.run(tf.local_variables_initializer())
            matchObj = re.search(r'-([0-9]*)\.meta', weight_path)
            start_step = int(matchObj.group(1))
            logging.info(start_step)
            saver.restore(sess=sess, save_path=weight_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for step in range(start_step, CFG.TRAIN_STEPS):
            _, cost, predict_ids, predict_scores, real_labels = sess.run([optimizer, loss, ids, scores, ground_labels], feed_dict={phase_tensor: 'train'})

            right = 0
            for n, sentence in enumerate(predict_ids):
                for m, wd in enumerate(sentence):
                    if wd == real_labels[n][m]:
                        right += 1
            train_accuracy = float(right)/(CFG.BATCH_SIZE * CFG.MAX_SEQ_LEN)

            train_summary = sess.run(train_summary_op_merge, feed_dict={accuracy: train_accuracy, phase_tensor: 'train'})

            summary_writer.add_summary(summary=train_summary, global_step=step)

            if step % CFG.DISPLAY_STEP == 0:
                logging.info('Step: {:d} cost= {:9f} train accuracy: {:9f}'.format(step + 1, cost, train_accuracy))

            if step % CFG.SAVE_STEP == 0:
                saver.save(sess, save_path=model_save_path, global_step=step)

        coord.request_stop()
        coord.join(threads=threads)
    sess.close()
    return

class Decoder(object):
    def __init__(self, filename):
        self.id_2_word = {}
        with open(filename, 'r') as reader:
            for i, line in enumerate(reader.readlines()):
                self.id_2_word[i] = line.strip().split()[0]

    def decoder(self, ids):
        words = []
        for id in ids:
            words.append(self.id_2_word[id])
        return ''.join(words)


    # with tf.Session() as sess:
    #     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #     sess.run(init_op)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     out_loss, out_ids, out_scores, images = sess.run([loss, ids, scores, train_images], feed_dict={phase_tensor:'train'})
    #     print(out_loss)
    #     print(out_ids)
    #     print(out_scores)
    #     img = Image.fromarray(images[0])
    #     img.show()
    #     coord.request_stop()
    #     coord.join(threads)

if __name__ == '__main__':
    train_atteion_network(CFG.DATASET_DIR)#, '/home/oushu/lixingyu/git_repo/attention_OCR/checkpoint/checkpoint0/attention_network_2018-12-25-11-49-34.ckpt-25300')