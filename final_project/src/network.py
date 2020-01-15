import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers import core as layers_core
import my_utils
from nltk.translate.bleu_score import sentence_bleu
from date_prepare import *

def cal_bleu(predict, result):
    predict = predict.tolist()
    # sequence_lengths = [len(seq) for seq in result]
    # max_sequence_length = max(sequence_lengths)
    # for sen in result:
    #     if len(sen) < max_sequence_length:
    #         lens = len(sen)
    #         for i in range(max_sequence_length - lens):
    #             sen.append(0)
    predict_list = []
    for i in range(len(predict)):
        try:
            end_ = predict[i].index(0)
        except:
            end_ = len(predict[i]) - 1
        if end_ < len(result[i]) - 1:
            predict_list.append(predict[i][:len(result[i])])
        else:
            predict_list.append(predict[i][:end_])
    total_score = 0
    for i in range(len(predict_list)):
        predict_sen = []
        result_sen = [[]]
        for j in range(len(predict_list[i])):
            predict_sen.append(target_id2word[predict_list[i][j]])
            # result_sen[0].append(target_id2word[result[i][j]])
        for j in range(len(result[i])):
            result_sen[0].append(target_id2word[result[i][j]])
        score = sentence_bleu(result_sen, predict_sen)
        total_score += score
    return total_score / len(predict)

class Seq2SeqModel():
    def __init__(self,
                 encoder_num_units,
                 decoder_num_units,
                 embedding_size,
                 num_layers,
                 source_vocab_size,
                 target_vocab_size,
                 batch_size,
                 attention=False,
                 beam_search=False,
                 beam_width=None,
                 mode=None):

        self.attention = attention
        self.beam_search = beam_search
        self.mode = mode

        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.encoder_num_units = encoder_num_units
        self.decoder_num_units = decoder_num_units

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = batch_size
        self.beam_width = beam_width

        self._make_graph()

    def _make_graph(self):

        self._init_placeholders()

        self._init_decoder_train_connectors()

        self._init_embedding()

        self._init_encoder()

        self._init_decoder()

        if self.mode == "Train":
            self._init_optimizer()

    def _init_placeholders(self):
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs'
        )

        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets',
        )

        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )

    def make_train_inputs(self, x, y):
        inputs, num_inputs = my_utils.prepare_batch(x)
        targets, num_targets = my_utils.prepare_batch(y)

        return {
            self.encoder_inputs: inputs,
            self.encoder_inputs_length: num_inputs,
            self.decoder_targets: targets,
            self.decoder_targets_length: num_targets
        }

    def make_infer_inputs(self, x):
        inputs, num_inputs = my_utils.prepare_batch(x)
        # 不知道target
        return {
            self.encoder_inputs: inputs,
            self.encoder_inputs_length: num_inputs
        }

    def _init_decoder_train_connectors(self):
        with tf.name_scope('DecoderTrainFeeds'):
            self.decoder_train_length = self.decoder_targets_length
            self.loss_weights = tf.ones(
                [self.batch_size, tf.reduce_max(self.decoder_train_length)],
                dtype=tf.float32)  # 权重初始为0

    def _init_embedding(self):
        self.embedding_encoder = tf.Variable(tf.random_uniform(
            [self.source_vocab_size,
             self.embedding_size]))
        self.encoder_embedding_inputs = tf.nn.embedding_lookup(
            self.embedding_encoder,
            self.encoder_inputs)

        self.embedding_decoder = tf.Variable(tf.random_uniform(
            [self.target_vocab_size,
             self.embedding_size]))

        self.decoder_embedding_inputs = tf.nn.embedding_lookup(
            self.embedding_decoder,
            self.decoder_targets)

    def _init_encoder(self):
        def make_cell(rnn_size):
            enc_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            return enc_cell

        with tf.variable_scope("Encoder") as scope:
            num_layers = self.num_layers
            #encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([make_cell(self.encoder_num_units) for _ in range(num_layers)])
            #encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([make_cell(self.encoder_num_units) for _ in range(num_layers)])
            encoder_cell_fw = make_cell(self.encoder_num_units)
            encoder_cell_bw = make_cell(self.encoder_num_units)
            # encoder_outputs:[batch_size,num_steps,rnn_size]
            # state 是每一层最后一个step的输出
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
                encoder_cell_fw,
                encoder_cell_bw,
                inputs=self.encoder_embedding_inputs,
                sequence_length=self.encoder_inputs_length,
                dtype=tf.float32,
            )
            self.encoder_outputs = tf.concat(
                [enc_outputs[0], enc_outputs[1]], -1)
            self.encoder_state = enc_state

    def _init_decoder(self):
        def make_cell(rnn_size):
            dec_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            return dec_cell

        def create_decoder_cell():
            cell = tf.contrib.rnn.MultiRNNCell(
                [make_cell(self.decoder_num_units) for _ in range(self.num_layers)])

            if self.beam_search and self.mode == "Infer":  # beam_search只用在测试阶段
                dec_start_state = seq2seq.tile_batch(
                    self.encoder_state, self.beam_width)  # 将encoder_state复制beam_width份
                # 将encoder_outputs复制beam_width份
                enc_outputs = seq2seq.tile_batch(
                    self.encoder_outputs, self.beam_width)
                # 将encoder_inputs_length复制beam_width份
                enc_lengths = seq2seq.tile_batch(
                    self.encoder_inputs_length, self.beam_width)
            else:
                dec_start_state = self.encoder_state  # encoder最后一个隐藏状态作为decoder的初始状态
                enc_outputs = self.encoder_outputs
                enc_lengths = self.encoder_inputs_length

            if self.attention:
                attention_states = enc_outputs

                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.decoder_num_units,
                    attention_states,
                    memory_sequence_length=enc_lengths)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell,
                    attention_mechanism,
                    attention_layer_size=self.decoder_num_units)

                if self.beam_search and self.mode == "Infer":
                    initial_state = decoder_cell.zero_state(
                        self.batch_size * self.beam_width, tf.float32)
                else:
                    initial_state = decoder_cell.zero_state(
                        self.batch_size, tf.float32)

                initial_state = initial_state.clone(cell_state=dec_start_state)
            else:
                initial_state = dec_start_state

            return decoder_cell, initial_state

        with tf.variable_scope("Decoder") as scope:
            projection_layer = layers_core.Dense(
                units=self.target_vocab_size, use_bias=False)  # use_bias
            #self.encoder_state = tuple(self.encoder_state[-1] for _ in range(self.num_layers))

            decoder_cell, initial_state = create_decoder_cell()

            if self.mode == "Train":
                training_helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_embedding_inputs,
                    self.decoder_train_length)

                training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=training_helper,
                    initial_state=initial_state,
                    output_layer=projection_layer)

                (self.decoder_outputs_train,
                 self.decoder_state_train,
                 final_sequence_length) = tf.contrib.seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    impute_finished=True,
                    scope=scope
                )

                self.decoder_logits_train = self.decoder_outputs_train.rnn_output
                decoder_predictions_train = tf.argmax(
                    self.decoder_logits_train, axis=-1)
                self.decoder_predictions_train = tf.identity(
                    decoder_predictions_train)

            elif self.mode == "Infer":
                start_tokens = tf.tile(tf.constant(
                    [2], dtype=tf.int32), [self.batch_size])

                if self.beam_search == True:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=decoder_cell,
                        embedding=self.embedding_decoder,
                        start_tokens=tf.ones_like(
                            self.encoder_inputs_length) * tf.constant(2, dtype=tf.int32),  # decoder第一个编码器的输入
                        #start_tokens = start_tokens,
                        end_token=tf.constant(0, dtype=tf.int32),  # 终止符
                        initial_state=initial_state,
                        beam_width=self.beam_width,
                        output_layer=projection_layer)

                    self.decoder_outputs_inference, __, ___ = tf.contrib.seq2seq.dynamic_decode(
                        decoder=inference_decoder,
                        maximum_iterations=tf.round(
                            tf.reduce_max(self.encoder_inputs_length)),
                        #maximum_iterations = 30,
                        impute_finished=False,
                        scope=scope)

                    self.decoder_predictions_inference = tf.identity(
                        self.decoder_outputs_inference.predicted_ids)

                else:
                    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        self.embedding_decoder,
                        start_tokens=start_tokens,
                        end_token=0)  # EOS id

                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=decoder_cell,
                        helper=inference_helper,
                        initial_state=initial_state,
                        output_layer=projection_layer)

                    self.decoder_outputs_inference, __, ___ = tf.contrib.seq2seq.dynamic_decode(
                        decoder=inference_decoder,
                        maximum_iterations=tf.round(
                            tf.reduce_max(self.encoder_inputs_length)) * 2,
                        impute_finished=False,
                        scope=scope)

                    self.decoder_predictions_inference = tf.identity(
                        self.decoder_outputs_inference.sample_id)

    def _init_optimizer(self):
        loss_mask = tf.sequence_mask(
            tf.to_int32(self.decoder_targets_length),
            tf.reduce_max(self.decoder_targets_length),
            dtype=tf.float32)

        self.loss = tf.contrib.seq2seq.sequence_loss(
            self.decoder_logits_train,
            self.decoder_targets,
            loss_mask)

        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()

        learning_rate = Learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var)
                            for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)





# tf.set_random_seed(1)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
source_vocab = list(set(source_vocab))
source_vocab_length = len(source_vocab)


def train():
    step = 0
    g = tf.Graph()
    with g.as_default():
        model = Seq2SeqModel(
            encoder_num_units=256,
            decoder_num_units=256,
            embedding_size=256,
            num_layers=2,
            source_vocab_size=source_vocab_length,
            target_vocab_size=len(target_id2word),
            batch_size=batch_size,
            attention=True,
            beam_search=True,
            beam_width=beam_width,
            mode="Train"
        )

        with tf.Session(config=tf.ConfigProto()) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            #summary_writer = tf.summary.FileWriter('../log', graph=sess.graph)

            for _epoch in range(1, batches_in_epoch + 1):
                for _batch in range(max_batches):
                    X, y = my_utils.input_generator(
                        train_source_id,
                        train_target_id,
                        batch_size)
                    feed_dict = model.make_train_inputs(
                        x=X,
                        y=y)
                    _, l, train_samples, summary_str, __ = sess.run(
                        [model.train_op,
                         model.loss,
                         model.decoder_predictions_train,
                         model.summary_op,
                         model.decoder_logits_train],
                        feed_dict)
                    #summary_writer.add_summary(summary_str, _epoch * _batch)
                    if step % 50 == 0:
                        bleu_score = cal_bleu(train_samples, y)
                        print('-' * 50)
                        print('-' * 50)
                        print('step {}'.format(step))
                        print('-' * 50)
                        print('minibatch loss: {}'.format(
                            sess.run(model.loss, feed_dict)))
                        print('-'*50)
                        print("Bleu Score: " + str(bleu_score))
                        print('-' * 50)
                        for i in range(5):
                            train_sentence = ''
                            for word in train_samples[i]:
                                train_sentence += target_id2word[word] + ' '
                            print('Predict Sen: ', end="")
                            print(train_sentence)
                        print('-' * 50)
                        for i in range(5):
                            result_sentence = ''
                            for word in y[i]:
                                result_sentence += target_id2word[word] + ' '
                            print('Target Sen: ', end="")
                            print(result_sentence)
                        print('-' * 50)
                        print('\n\n\n')
                    # if step % 250 == 0:
                    #     dev_x , dev_y = my_utils.dev_input_generator(dev_source_id,dev_target_id)
                    #     feed_dict = model.make_train_inputs(
                    #         x=X,
                    #         y=y)
                    step += 1

                print(_epoch, 'epoch finished')

                if _epoch % epoch_be_saved == 0:
                    saver.save(sess, '../models/' +
                               'nmt.ckpt', global_step=step)
                    print('model saved.')

            print('finish training')

if  __name__ == "__main__":

    step = 0
    batch_size = 32
    max_batches = int(len(train_source) / batch_size)
    batches_in_epoch = 1000
    epoch_be_saved = 5
    beam_width = 2
    Learning_rate = 0.01

    train()
    checkpoint = '../models/' + 'nmt.ckpt-' + '3750'

    batch_size = 20
    beam_width = 2
    # target_vocab = list(set(target_vocab))
    # target_vocab_length = len(target_vocab)
    # print(target_vocab_length)
    infer_graph = tf.Graph()
    with infer_graph.as_default():
        model = Seq2SeqModel(
            encoder_num_units=256,
            decoder_num_units=256,
            embedding_size=256,
            num_layers=2,
            source_vocab_size=source_vocab_length,
            target_vocab_size=len(target_id2word),
            batch_size=batch_size,
            attention=True,
            beam_search=True,
            beam_width=beam_width,
            mode="Infer"
        )
        print('model constructed.')

        with tf.Session() as sess:
            print("*********************************************************")
            #         loader = tf.train.import_meta_graph(checkpoint + '.meta')
            #         loader.restore(sess, checkpoint)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)
            print("*********************************************************")
            for _batch in range(50):
                print("*********************************************************")
                X, y = my_utils.test_input_generator(
                    test_source_id,
                    test_target_id,
                    batch_size)

                feed_dict = model.make_infer_inputs(X)
                inf_logits = model.decoder_predictions_inference
                translations = sess.run(inf_logits, feed_dict)
                print("*********************************************************")
                if model.beam_search == True:
                    print("*********************************************************")
                    for i, translation in enumerate(translations):
                        print(i)
                        for b in range(beam_width):
                            sent = ""
                            for words in translation:
                                sent += target_id2word[words[b]] + ' '
                            print(sent, '\n')
                else:
                    for translation in translations:
                        sent = ""
                        for word in translation:
                            sent += target_id2word[word] + ' '
                        print(sent, '\n')

