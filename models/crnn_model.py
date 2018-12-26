import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
sys.path.append('/home/oushu/lixingyu/git_repo/attention_OCR')

from tensorflow.contrib import rnn

from models import cnn_basenet
from config.global_config import CFG
import numpy as np


class ShadowNet(cnn_basenet.CNNBaseModel):
    def __init__(self, phase):
        super(ShadowNet, self).__init__()
        # self._phase = phase
        # # self._hidden_nums = hidden_nums
        # # self._layers_nums = layers_nums
        # # self._seq_length = seq_length
        # self._num_classes = num_classes
        # # self._rnn_cell_type = rnn_cell_type.lower()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._is_training = tf.equal(self._train_phase, phase)
        # if self._rnn_cell_type not in ['lstm', 'gru']:
        #     raise ValueError('rnn_cell_type should be in [\'lstm\', \'gru\']')
        # return

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        if not isinstance(value, str):
            raise TypeError('value should be a str \'Test\' or \'Train\'')
        if value.lower() not in ['test', 'train']:
            raise ValueError('value should be a str \'Test\' or \'Train\'')
        self._phase = value.lower()
        return

    def _conv_stage(self, inputdata, out_dims, name=None):
        conv = self.conv2d(inputdata=inputdata, out_channel=out_dims,
                           kernel_size=3, stride=1, use_bias=False,
                           name=name)
        relu = self.relu(inputdata=conv)
        max_pool = self.maxpooling(inputdata=relu, kernel_size=2, stride=2)
        return max_pool

    def _feature_sequence_extraction(self, inputdata):
        tensor_dict = dict()

        conv1 = self._conv_stage(inputdata=inputdata, out_dims=64, name='conv1')  # batch*16*50*64
        tensor_dict['conv1'] = conv1

        conv2 = self._conv_stage(inputdata=conv1, out_dims=128, name='conv2')  # batch*8*25*128
        tensor_dict['conv2'] = conv2

        conv3 = self.conv2d(inputdata=conv2, out_channel=256,
                            kernel_size=3, stride=1, use_bias=False,
                            name='conv3')  # batch*8*25*256
        relu3 = self.relu(conv3)  # batch*8*25*256
        tensor_dict['conv3'] = conv3
        tensor_dict['relu3'] = relu3

        conv4 = self.conv2d(inputdata=relu3, out_channel=256,
                            kernel_size=3, stride=1, use_bias=False,
                            name='conv4')  # batch*8*25*256
        relu4 = self.relu(conv4)  # batch*8*25*256
        max_pool4 = self.maxpooling(inputdata=relu4, kernel_size=[2, 1], stride=[2, 1],
                                    padding='VALID')  # batch*4*25*256
        tensor_dict['conv4'] = conv4
        tensor_dict['relu4'] = relu4
        tensor_dict['max_pool4'] = max_pool4

        conv5 = self.conv2d(inputdata=max_pool4, out_channel=512,
                            kernel_size=3, stride=1, use_bias=False,
                            name='conv5')  # batch*4*25*512
        conv5_bn5 = self.layerbn(inputdata=conv5, is_training=self._is_training, name='bn5')
        relu5 = self.relu(conv5_bn5)  # batch*4*25*512
        tensor_dict['conv5'] = conv5
        tensor_dict['relu5'] = relu5
        tensor_dict['bn5'] = conv5_bn5

        conv6 = self.conv2d(inputdata=relu5, out_channel=512,
                            kernel_size=3, stride=1, use_bias=False,
                            name='conv6')  # batch*4*25*512
        conv6_bn6 = self.layerbn(inputdata=conv6, is_training=self._is_training, name='bn6')
        relu6 = self.relu(conv6_bn6)  # batch*4*25*512
        max_pool6 = self.maxpooling(inputdata=relu6,
                                    kernel_size=[2, 1], stride=[2, 1])  # batch*2*25*512
        tensor_dict['conv6'] = conv6
        tensor_dict['relu6'] = relu6
        tensor_dict['bn6'] = conv6_bn6
        tensor_dict['max_pool6'] = max_pool6

        conv7 = self.conv2d(inputdata=max_pool6, out_channel=512,
                            kernel_size=2, stride=[2, 1], use_bias=False,
                            name='conv7')  # batch*1*25*512
        relu7 = self.relu(conv7)  # batch*1*25*512
        tensor_dict['conv7'] = conv7
        tensor_dict['relu7'] = relu7
        return relu7, tensor_dict

    def _map_to_sequence(self, inputdata):
        shape = inputdata.get_shape().as_list()
        assert shape[1] == 1  # H of the feature map must equal to 1
        return self.squeeze(inputdata=inputdata, axis=1)

    def build_shadownet(self, inputdata, labels):
        with tf.variable_scope('cnn_subnetwork'):
            # print([labels.shape.as_list()[0], CFG.MAX_SEQ_LEN])
            if labels:
                labels = tf.sparse_to_dense(labels.indices, [CFG.BATCH_SIZE, CFG.MAX_SEQ_LEN], labels.values)
                labels_one_hot = slim.one_hot_encoding(labels, num_classes=CFG.CLASSES_NUMS)
            else:
                labels_one_hot = None
            cnn_out, tensor_dict = self._feature_sequence_extraction(inputdata=inputdata)

            sequence = self._map_to_sequence(inputdata=cnn_out)
            net_out = self.encode_coordinate_fn(sequence)
            sequence_logit = self.sequence_logit_fn(net_out, labels_one_hot)
            ids, logit_prob, scores = self.get_char_prdict(sequence_logit)
            loss = self.create_loss(labels, logit_prob)
            return loss, ids, scores, tensor_dict

    def encode_coordinate_fn(self, net):
        batch_size, w, _ = net.shape.as_list()
        x = tf.range(w)
        w_loc = slim.one_hot_encoding(x, num_classes=w)
        loc = tf.tile(tf.expand_dims(w_loc, 0), [batch_size, 1, 1])
        return tf.concat([net, loc], 2)

    def sequence_logit_fn(self, net, labels_one_hot):
        sequence_layer = AttentionWithAutoRegression(net, labels_one_hot)
        return sequence_layer.create_logits()

    def get_char_prdict(self, chars_logit):
        with tf.variable_scope('log_probabilities'):
            reduction_indices = len(chars_logit.shape.as_list())-1
            max_logits = tf.reduce_max(
                chars_logit, reduction_indices=reduction_indices, keepdims=True)
            safe_logits = tf.subtract(chars_logit, max_logits)
            sum_exp = tf.reduce_sum(
                tf.exp(safe_logits),
                reduction_indices=reduction_indices,
                keepdims=True
            )
            log_probs = tf.subtract(safe_logits, tf.log(sum_exp))
        ids = tf.to_int32(tf.argmax(log_probs, axis=2), name='predicted_chars')
        mask = tf.cast(
            slim.one_hot_encoding(ids, CFG.CLASSES_NUMS), tf.bool
        )
        all_scores = tf.nn.softmax(chars_logit)
        selected_scores = tf.boolean_mask(all_scores, mask, name='char_scores')
        scores = tf.reshape(selected_scores, shape=(-1, CFG.MAX_SEQ_LEN))
        return ids, log_probs, scores

    def sequence_loss_fn(self, chars_logits, chars_labels):
        with tf.variable_scope('sequence_loss_fn/SLF'):
            labels_list = tf.unstack(chars_labels, axis=1)
            batch_size, seq_length, _ = chars_logits.shape.as_list()
            reject_char = tf.constant(
                0,
                shape=(batch_size, seq_length),
                dtype=tf.int32
            )
            known_char = tf.not_equal(chars_labels, reject_char)
            weights = tf.to_float(known_char)
            logits_list = tf.unstack(chars_logits, axis=1)
            weights_list = tf.unstack(weights, axis=1)
            loss = tf.contrib.legacy_seq2seq.sequence_loss(
                logits_list,
                labels_list,
                weights_list,
                softmax_loss_function=get_softmax_fn(),
                average_across_timesteps=False
            )
            tf.losses.add_loss(loss)
            return loss


    def create_loss(self, labels, chars_logit):
        self.sequence_loss_fn(chars_logit, labels)
        total_loss = tf.losses.get_total_loss()
        return total_loss


def get_softmax_fn():
    def loss_fn(labels, logits):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    return loss_fn

def arr2sparse(arr_tensor):
    arr_idx = tf.where(tf.not_equal(arr_tensor, 0))
    arr_sparse = tf.SparseTensor(arr_idx, tf.gather_nd(arr_tensor, arr_idx), arr_tensor.get_shape())
    return arr_sparse


class AttentionWithAutoRegression(object):
    def __init__(self, net, labels_one_hot):
        self.net = net
        self.batch_size = self.net.get_shape().dims[0].value
        self.zero_labels = tf.zeros([self.batch_size, CFG.CLASSES_NUMS])
        self.labels_one_hot = labels_one_hot

        self.char_logits = {}
        regularizer = slim.l2_regularizer(0.0)
        self.softmax_w = slim.model_variable(
            'softmax_w',
            [CFG.NUM_LSTM_UNITS, CFG.CLASSES_NUMS],
            initializer=orthogonal_initializer,
            regularizer=regularizer
        )
        self.softmax_b = slim.model_variable(
            'softmax_b',
            [CFG.CLASSES_NUMS],
            initializer=tf.zeros_initializer(),
            regularizer=regularizer
        )

    def char_logits_fn(self, inputs, char_index):
        if char_index not in self.char_logits:
            self.char_logits[char_index] = tf.nn.xw_plus_b(inputs, self.softmax_w, self.softmax_b)
        return self.char_logits[char_index]

    def char_one_hot(self, logit):
        prediction = tf.argmax(logit, axis=1)
        return slim.one_hot_encoding(prediction, CFG.CLASSES_NUMS)

    def get_train_input(self, prev, i):
        if i == 0:
            return self.zero_labels
        else:
            return self.labels_one_hot[:, i-1, :]

    def get_eval_input(self, prev, i):
        if i == 0:
            return self.zero_labels
        else:
            logit = self.char_logits_fn(prev, char_index=i-1)
            return self.char_one_hot(logit)

    def get_input(self, prev, i):
        if self.labels_one_hot is not None:
            return self.get_train_input(prev, i)
        else:
            return self.get_eval_input(prev, i)

    def create_logits(self):
        with tf.variable_scope('LSTM'):
            first_label = self.get_input(prev=None, i=0)
            decoder_inputs = [first_label] + [None] * (CFG.MAX_SEQ_LEN-1)
            lstm_cell = tf.contrib.rnn.LSTMCell(
                CFG.NUM_LSTM_UNITS,
                use_peepholes=False,
                cell_clip=CFG.LSTM_STATE_CLIP_VAL,
                state_is_tuple=True,
                initializer=orthogonal_initializer
            )
            lstm_outputs, _ = tf.contrib.legacy_seq2seq.attention_decoder(
                decoder_inputs=decoder_inputs,
                initial_state=lstm_cell.zero_state(self.batch_size, tf.float32),
                attention_states=self.net,
                loop_function=self.get_input,
                cell=lstm_cell
            )
        with tf.variable_scope('logits'):
            logits_list = [
                tf.expand_dims(self.char_logits_fn(logit, i), axis=1)
                for i, logit in enumerate(lstm_outputs)
            ]
        return tf.concat(logits_list, 1)

    # def unroll_cell(self, decoder_input):

def orthogonal_initializer(shape, dtype=tf.float32, *args, **kwargs):
    del args
    del kwargs
    flat_shape = (shape[0], np.prod(shape[1:]))
    w = np.random.randn(*flat_shape)
    u, _, v = np.linalg.svd(w, full_matrices=False)
    w = u if u.shape == flat_shape else v
    return tf.constant(w.reshape(shape), dtype=dtype)

    # def _sequence_label(self, inputdata):
    #     """
    #     Implement the sequence label part of the network
    #     :param inputdata:
    #     :return:
    #     """
    #     if self._rnn_cell_type == 'lstm':
    #         with tf.variable_scope('LSTMLayers'):
    #             # construct stack lstm rcnn layer
    #             # forward lstm cell
    #             fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in
    #                             [self._hidden_nums, self._hidden_nums]]
    #             # Backward direction cells
    #             bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in
    #                             [self._hidden_nums, self._hidden_nums]]
    #
    #             stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
    #                 fw_cell_list, bw_cell_list, inputdata, dtype=tf.float32)
    #
    #             def f1():
    #                 """
    #
    #                 :return:
    #                 """
    #                 return self.dropout(inputdata=stack_lstm_layer, keep_prob=0.5)
    #
    #             def f2():
    #                 """
    #
    #                 :return:
    #                 """
    #                 return stack_lstm_layer
    #
    #             stack_lstm_layer = tf.cond(self._is_training, f1, f2)
    #
    #             # [batch, width, 2*n_hidden]
    #             [batch_s, _, hidden_nums] = inputdata.get_shape().as_list()
    #
    #             # [batch x width, 2*n_hidden]
    #             rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, hidden_nums])
    #
    #             var_w = tf.Variable(tf.truncated_normal([hidden_nums, self._num_classes],
    #                                                     stddev=0.1), name="w")
    #
    #             # Doing the affine projection
    #             # logits = tf.matmul(rnn_reshaped, var_w)
    #             logits = slim.fully_connected(inputs=rnn_reshaped, num_outputs=self._num_classes,
    #                                           activation_fn=None)
    #
    #             logits = tf.reshape(logits, [batch_s, -1, self._num_classes])
    #
    #             # raw_pred = tf.argmax(tf.nn.softmax(logits),
    #             #                      axis=2, name='raw_prediction')
    #
    #             # Swap batch and batch axis
    #             rnn_out = tf.transpose(logits, (1, 0, 2),
    #                                    name='transpose_time_major')  # [width, batch, n_classes]
    #     else:
    #         with tf.variable_scope('GRULayers'):
    #             # construct stack fru rcnn layer
    #             # forward gru cell
    #             fw_cell_list = [rnn.GRUCell(nh) for nh in
    #                             [self._hidden_nums, self._hidden_nums]]
    #             # Backward direction cells
    #             bw_cell_list = [rnn.GRUCell(nh) for nh in
    #                             [self._hidden_nums, self._hidden_nums]]
    #
    #             stack_gru_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
    #                 fw_cell_list, bw_cell_list, inputdata, dtype=tf.float32)
    #
    #             def f3():
    #                 """
    #
    #                 :return:
    #                 """
    #                 return self.dropout(inputdata=stack_gru_layer, keep_prob=0.5)
    #
    #             def f4():
    #                 """
    #
    #                 :return:
    #                 """
    #                 return stack_gru_layer
    #
    #             stack_gru_layer = tf.cond(self._is_training, f3, f4)
    #
    #             # [batch, width, 2*n_hidden]
    #             [batch_s, _, hidden_nums] = inputdata.get_shape().as_list()
    #
    #             # [batch x width, 2*n_hidden]
    #             rnn_reshaped = tf.reshape(stack_gru_layer, [-1, hidden_nums])
    #
    #             var_w = tf.Variable(tf.truncated_normal([hidden_nums, self._num_classes],
    #                                                     stddev=0.1),
    #                                 name="w")
    #
    #             # Doing the affine projection
    #             # logits = tf.matmul(rnn_reshaped, var_w)
    #             logits = slim.fully_connected(inputs=rnn_reshaped, num_outputs=self._num_classes,
    #                                           activation_fn=None)
    #
    #             logits = tf.reshape(logits, [batch_s, -1, self._num_classes])
    #
    #             # raw_pred = tf.argmax(tf.nn.softmax(logits),
    #             #                      axis=2, name='raw_prediction')
    #
    #             # Swap batch and batch axis
    #             rnn_out = tf.transpose(logits, (1, 0, 2),
    #                                    name='transpose_time_major')  # [width, batch, n_classes]
    #
    #     return rnn_out

    # def build_shadownet_cnn_subnet(self, inputdata):
    #     """
    #     Build the cnn feature extraction part of the crnn model used for classification
    #     :param inputdata:
    #     :return:
    #     """
    #     # first apply the cnn feture extraction stage
    #     with tf.variable_scope('cnn_subnetwork'):
    #         cnn_out = self._feature_sequence_extraction(inputdata=inputdata)
    #
    #         fc1 = self.fullyconnect(inputdata=cnn_out, out_dim=4096, use_bias=False, name='fc1')
    #
    #         relu1 = self.relu(inputdata=fc1, name='relu1')
    #
    #         fc2 = self.fullyconnect(inputdata=relu1, out_dim=CFG.TRAIN.CLASSES_NUMS,
    #                                 use_bias=False, name='fc2')
    #
    #     return fc2

if __name__ == '__main__':
    label = tf.ones([32, 10], tf.int64)
    label = arr2sparse(label)
    x = tf.ones([32, 32, 100, 3], tf.float32)
    network = ShadowNet('train')
    loss, ids, score, _ = network.build_shadownet(x, label)
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

    #     l, id, s = sess.run([loss, ids, score])
    # print(l)
    # print(id)
    # print(s)