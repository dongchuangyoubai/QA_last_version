import logging
import tensorflow as tf
from tensorflow.python.ops import array_ops

from attention_wrapper import *
from data_utils import *
from general_utils import Progbar

def _reverse(input_, seq_lengths, seq_axis, batch_axis):
    if seq_lengths is not None:
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_axis=seq_axis, batch_axis=batch_axis)
    else:
        return array_ops.reverse(input_, axis=[seq_axis])


class Encoder(object):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def encode(self, inputs, masks):
        """
        :param inputs: vector representations of question and passage (a tuple)
        :param masks: masking sequences for both question and passage (a tuple)
        :return: an encoded representation of the question and passage.
        """
        question, passage = inputs
        masks_question, masks_passage = masks

        with tf.variable_scope('encoder_question'):
            question_lstm = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            encoded_question, (q_rep, _) = tf.nn.dynamic_rnn(question_lstm, question, masks_question, dtype=tf.float32)

        with tf.variable_scope('encoder_passage'):
            passage_lstm = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            encoded_passage, (p_rep, _) = tf.nn.dynamic_rnn(passage_lstm, passage, masks_passage, dtype=tf.float32)

        return encoded_question, encoded_passage, q_rep, p_rep


class Decoder(object):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def run_match_lstm(self, encoder_rep, masks):
        encoder_question, encoder_passage = encoder_rep
        masks_question, masks_passage = masks

        def match_lstm_cell_attention_fn(curr_input, state):
            return tf.concat([curr_input, state], axis=-1)

        query_depth = encoder_question.get_shape()[-1]

        # output attention is false because we want to output the cell output and not the attention values
        with tf.variable_scope("match_lstm_attention"):
            attention_mechanism_match_lstm = BahdanauAttention(query_depth, encoder_question,
                                                               memory_sequence_length=masks_question)
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            lstm_attention = AttentionWrapper(cell, attention_mechanism_match_lstm, output_attention=False,
                                              attention_input_fn=match_lstm_cell_attention_fn)

            # we don't mask the passage because masking the memories will be handled by the pointerNet
            reverse_encoder_passage = _reverse(encoder_passage, masks_passage, 1, 0)
            output_attention_fw, _ = tf.nn.dynamic_rnn(lstm_attention, encoder_passage, dtype=tf.float32)
            output_attention_bw, _ = tf.nn.dynamic_rnn(lstm_attention, reverse_encoder_passage, dtype=tf.float32)

            output_attention_bw = _reverse(output_attention_bw, masks_passage, 1, 0)

        output_attention = tf.concat([output_attention_fw, output_attention_bw], axis=-1)  # (-1, P, 2*H)
        return output_attention

    def run_answer_ptr(self, output_attention, masks, labels):
        masks_question, masks_passage = masks
        labels = tf.unstack(labels, axis=1)

        def answer_ptr_cell_input_fn(curr_input, context):
            return context  # independent of question

        query_depth_answer_ptr = output_attention.get_shape()[-1]

        with tf.variable_scope("answer_ptr_attention"):
            attention_mechanism_answer_ptr = BahdanauAttention(query_depth_answer_ptr, output_attention,
                                                               memory_sequence_length=masks_passage)
            # output attention is true because we want to output the attention values
            cell_answer_ptr = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            answer_ptr_attention = AttentionWrapper(cell_answer_ptr, attention_mechanism_answer_ptr,
                                                    cell_input_fn=answer_ptr_cell_input_fn)
            logits, _ = tf.nn.static_rnn(answer_ptr_attention, labels, dtype=tf.float32)

        return logits

    def decode(self, encoder_rep, masks, labels):
        """
        takes in encoded_rep
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param encoded_rep:
        :param masks
        :param labels


        :return: logits: for each word in passage the probability that it is the start word and end word.
        """

        output_attender = self.run_match_lstm(encoder_rep, masks)
        logits = self.run_answer_ptr(output_attender, masks, labels)

        return logits


class QASystem(object):
    def __init__(self, encoder, decoder, pretrained_embeddings, config):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        # ==== set up logging ======
        logger = logging.getLogger("QASystemLogger")
        self.logger = logger

        # ==== set up placeholder tokens ========
        self.embeddings = pretrained_embeddings
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        self.setup_placeholders()
        # ==== assemble pieces ====
        with tf.variable_scope("qa"):
            self.setup_word_embeddings()
            self.setup_system()
            self.setup_loss()
            self.setup_train_op()
            self.saver = tf.train.Saver()

    def setup_placeholders(self):
        self.question_ids = tf.placeholder(tf.int32, shape=[None, None], name="question_ids")
        self.passage_ids = tf.placeholder(tf.int32, shape=[None, None], name="passage_ids")

        self.question_lengths = tf.placeholder(tf.int32, shape=[None], name="question_lengths")
        self.passage_lengths = tf.placeholder(tf.int32, shape=[None], name="passage_lengths")

        self.labels = tf.placeholder(tf.int32, shape = [None, 2], name="gold_labels")
        self.dropout = tf.placeholder(tf.float32, shape=[], name="dropout")

    def setup_word_embeddings(self):
        """
        Create an embedding matrix (initialised with pretrained glove vectors and updated only if self.config.train_embeddings is true)
        lookup into this matrix and apply dropout (which is 1 at test time and self.config.dropout at train time)
        :return:
        """
        with tf.variable_scope("vocab_embeddings"):
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32,
                                           trainable=self.config.train_embeddings)
            question_emb = tf.nn.embedding_lookup(_word_embeddings, self.question_ids, name="question")  # (-1, Q, D)
            passage_emb = tf.nn.embedding_lookup(_word_embeddings, self.passage_ids, name="passage")  # (-1, P, D)
            # Apply dropout
            self.question = tf.nn.dropout(question_emb, self.dropout)
            self.passage = tf.nn.dropout(passage_emb, self.dropout)

    def setup_system(self):
        """
           Apply the encoder to the question and passage embeddings. Follow that up by Match-LSTM and Answer-Ptr
        """
        encoder = self.encoder
        decoder = self.decoder
        encoded_question, encoded_passage, q_rep, p_rep = encoder.encode([self.question, self.passage],
                                                                         [self.question_lengths, self.passage_lengths])

        self.logger.info("\n========Using Match LSTM=========\n")
        self.logits = decoder.decode([encoded_question, encoded_passage],
                                [self.question_lengths, self.passage_lengths], self.labels)


    def setup_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step"):
            adam_optimizer = tf.train.AdamOptimizer()
            grads, vars = zip(*adam_optimizer.compute_gradients(self.loss))

            clip_val = self.config.max_gradient_norm
            # if -1 then do not perform gradient clipping
            if clip_val != -1:
                clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.max_gradient_norm)
                self.global_grad = tf.global_norm(clipped_grads)
                self.gradients = zip(clipped_grads, vars)
            else:
                self.global_grad = tf.global_norm(grads)
                self.gradients = zip(grads, vars)

            self.train_op = adam_optimizer.apply_gradients(self.gradients)

        self.init = tf.global_variables_initializer()

    def get_feed_dict(self, questions, contexts, answers, dropout_val):
        """
        -arg questions: A list of list of ids representing the question sentence
        -arg contexts: A list of list of ids representing the context paragraph
        -arg dropout_val: A float representing the keep probability for dropout

        :return: dict {placeholders: value}
        """

        questions = [list(x) for x in questions]
        contexts = [list(x) for x in contexts]
        answers = [list(x) for x in answers]

        padded_questions, question_lengths = pad_sequences(questions, 0)
        padded_contexts, passage_lengths = pad_sequences(contexts, 0)

        feed = {
            self.question_ids: padded_questions,
            self.passage_ids: padded_contexts,
            self.question_lengths: question_lengths,
            self.passage_lengths: passage_lengths,
            self.labels: answers,
            self.dropout: dropout_val
        }

        return feed

    def setup_loss(self):
        """
        self.logits are the 2 sets of logit (num_classes) values for each example, masked with float(-inf) beyond the true sequence length
        :return: Loss for the current batch of examples
        """

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[0], labels=self.labels[:, 0])
        losses += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[1], labels=self.labels[:, 1])
        self.loss = tf.clip_by_value(tf.reduce_mean(losses), 1e-8, 100)

    def initialize_model(self, session, train_dir):
        """
            param: session managed from train.py
            param: train_dir : the directory in which models are saved

        """
        ckpt = tf.train.get_checkpoint_state(train_dir)
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            self.logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            self.logger.info("Created model with fresh parameters.")
            session.run(self.init)
            self.logger.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))

    def test(self, session, valid):
        """
        valid: a list containing q, c and a.
        :return: loss on the valid dataset and the logit values
        """
        q, c, a = valid
        # at test time we do not perform dropout.
        input_feed = self.get_feed_dict(q, c, a, 1.0)
        output_feed = [self.logits]
        outputs = session.run(output_feed, input_feed)

        return outputs[0][0], outputs[0][1]

    def answer(self, session, dataset):
        """
        Get the answers for dataset. Independent of how data iteration is implemented
        :param session:
        :param dataset:
        :return:
        """

        yp, yp2 = self.test(session, dataset)

        # -- Boundary Model with a max span restriction of 15

        def func(y1, y2):
            max_ans = -999999
            a_s, a_e = 0, 0
            num_classes = len(y1)
            for i in range(num_classes):
                for j in range(15):
                    if i + j >= num_classes:
                        break

                    curr_a_s = y1[i]
                    curr_a_e = y2[i + j]
                    if (curr_a_e + curr_a_s) > max_ans:
                        max_ans = curr_a_e + curr_a_s
                        a_s = i
                        a_e = i + j
            return a_s, a_e

        a_s, a_e = [], []
        for i in range(yp.shape[0]):
            _a_s, _a_e = func(yp[i], yp2[i])
            a_s.append(_a_s)
            a_e.append(_a_e)

        return np.array(a_s), np.array(a_e)

    def evaluate_model(self, session, dataset):
        """
        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :return: exact match scores
        """

        q, c, a = zip(*[[_q, _c, _a] for (_q, _c, _a) in dataset])

        sample = len(dataset)
        a_s, a_o = self.answer(session, [q, c, a])
        answers = np.hstack([a_s.reshape([sample, -1]), a_o.reshape([sample, -1])])
        gold_answers = np.array([a for (_, _, a) in dataset])

        em_score = 0
        em_1 = 0
        em_2 = 0
        for i in range(sample):
            gold_s, gold_e = gold_answers[i]
            s, e = answers[i]
            if (s == gold_s): em_1 += 1.0
            if (e == gold_e): em_2 += 1.0
            if (s == gold_s and e == gold_e):
                em_score += 1.0

        em_1 /= float(len(answers))
        em_2 /= float(len(answers))
        self.logger.info("\nExact match on 1st token: %5.4f | Exact match on 2nd token: %5.4f\n" % (em_1, em_2))

        em_score /= float(len(answers))

        return em_score

    def run_epoch(self, session, train):
        """
        Perform one complete pass over the training data and evaluate on dev
        """
        nbatches = (len(train) + self.config.batch_size - 1) / self.config.batch_size
        prog = Progbar(target=nbatches)

        for i, (q_batch, c_batch, a_batch) in enumerate(minibatches(train, self.config.batch_size)):
            # at training time, dropout needs to be on.
            input_feed = self.get_feed_dict(q_batch, c_batch, a_batch, self.config.dropout_val)

            _, train_loss = session.run([self.train_op, self.loss], feed_dict=input_feed)
            prog.update(i + 1, [("train loss", train_loss)])
            summary = session.run(self.merged, input_feed)
            self.writer.add_summary(summary, i)

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        :param session: it should be passed in from train.py
        :param dataset: a list containing the training and dev data
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        if not tf.gfile.Exists(train_dir):
            tf.gfile.MkDir(train_dir)

        train, dev = dataset

        em = self.evaluate_model(session, dev)
        self.logger.info("\n#-----------Initial Exact match on dev set: %5.4f ---------------#\n" % em)
        # self.logger.info("#-----------Initial F1 on dev set: %5.4f ---------------#" %f1)

        best_em = 0
        import datetime

        tf.summary.scalar('Loss', self.loss)
        tf.summary.scalar('Accuracy', em)
        self.merged = tf.summary.merge_all()
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        self.writer = tf.summary.FileWriter(logdir, session.graph)

        for epoch in range(self.config.num_epochs):
            self.logger.info("\n*********************EPOCH: %d*********************\n" % (epoch + 1))
            self.run_epoch(session, train)
            em = self.evaluate_model(session, dev)
            self.logger.info("\n#-----------Exact match on dev set: %5.4f #-----------\n" % em)
            # self.logger.info("#-----------F1 on dev set: %5.4f #-----------" %f1)

            # ======== Save model if it is the best so far ========
            if (em > best_em):
                self.saver.save(session, "%s/best_model.chk" % train_dir)
                best_em = em