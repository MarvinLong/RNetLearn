import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net


class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True):
        # get the init paras
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)
        self.char_mat = tf.get_variable(
            "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        N, CL = config.batch_size, config.char_limit
        self.c_maxlen = tf.reduce_max(self.c_len)
        self.q_maxlen = tf.reduce_max(self.q_len)
        self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
        self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
        self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
        self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
        self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
        self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
        self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
        self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])

        self.ch_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
        self.qh_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

    # build the model
        config = self.config
        N, PL, QL, CL, d, dc, dg = config.batch_size, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.char_hidden
    # define the gru
        gru = cudnn_gru if config.use_cudnn else native_gru

        # --- 1. question and passage encoder
        with tf.variable_scope('emb'):
            # get the char embedding
            with tf.variable_scope('char'):
                ch_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.ch), [N * PL, CL, dc])
                qh_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.qh), [N * QL, CL, dc])
                ch_emb = dropout(
                    ch_emb, keep_prob=config.keep_prob, is_train=self.is_train)
                qh_emb = dropout(
                    qh_emb, keep_prob=config.keep_prob, is_train=self.is_train)
                cell_fw = tf.contrib.rnn.GRUCell(dg)
                cell_bw = tf.contrib.rnn.GRUCell(dg)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, ch_emb, self.ch_len, dtype=tf.float32)
                ch_emb = tf.concat([state_fw, state_bw], axis=1)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, qh_emb, self.qh_len, dtype=tf.float32)
                qh_emb = tf.concat([state_fw, state_bw], axis=1)
                qch_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
                pch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])

            # get the word embedding
            with tf.variable_scope('word'):
                pw_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                qw_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

        # concat the char and word embedding
        p_emb = tf.concat([pw_emb, pch_emb], axis=2)
        q_emb = tf.concat([qw_emb, qch_emb], axis=2)

        # is this used the
        with tf.variable_scope('encoder'):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=p_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            u_p = rnn(p_emb, seq_len=self.c_len)
            u_q = rnn(q_emb, seq_len=self.q_len)

        # --- 2. gated attenction-based rnn
        with tf.variable_scope('qp_matching'):
            # do attention and use gate
            qp_att = dot_attention(u_p, u_q, mask=self.q_mask, hidden=d,
                                   keep_prob=config.keep_prob, is_train=self.is_train)
            # get the v_p
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qp_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            v_p = rnn(qp_att, seq_len=self.c_len)

        # --- 3. self-match attention 
        with tf.variable_scope('self_matching'):
            self_att = dot_attention(v_p, v_p, mask=self.c_mask, hidden=d,
                                     keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            h_p = rnn(self_att, seq_len=self.c_len)

        # --- 4. output layer
        with tf.variable_scope('ouput'):
            with tf.variable_scope('pointer'):
                # how to calculate r^Q with pooling ???
                init = summ(u_q[:, :, -2 * d:], d, mask=self.q_mask,
                            keep_prob=config.ptr_keep_prob, is_train=self.is_train)
                pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
                )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
                logits1, logits2 = pointer(init, h_p, d, self.c_mask)

            with tf.variable_scope('predict'):
                outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                                  tf.expand_dims(tf.nn.softmax(logits2), axis=1))
                outer = tf.matrix_band_part(outer, 0, 15)
                self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
                self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits1, labels=tf.stop_gradient(self.y1))
                losses2 = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits2, labels=tf.stop_gradient(self.y2))
                self.loss = tf.reduce_mean(losses + losses2)

        if trainable:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
