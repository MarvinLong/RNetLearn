import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net
class R-NET():
	def __inint__(self,config,wrod_mat=None,char_mat=None,is_train=True):
	# build the model
		# define the gru
		gru = cudnn_gru if config.use_cudnn else native_gru
		#--- 1. question and passage encoder
		with tf.variable_scope('emb'):
			# get the char embedding
			with tf.variable_scope('char'):

			# get the word embedding
			with tf.variable_scope('word'):
		
		# concat the char and word embedding
		p_emb = tf.concat([pw_emb,pch_emb])
		q_emb = tf.concat([qw_emb,qch_emb])

		# is this used the bi-rnn ???
		with tf.variable_scope('encoder'):
			rnn = gru(num_layers=3,num_units=d, batch_size=N, input_size=c_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            u_p = rnn(p_emb,seq_len=)
            u_q = rnn(q_emb,seq_len=)

		#--- 2. gated attenction-based rnn
		with tf.variable_scope('qp_matching'):
			# do attention and use gate
			qp_att = dot_attention(u_p,u_q,mask=self.q_mask,)
			# get the v_p
			rnn = gru(,,,input_size=qp_att.get_shape().as_list()[-1],)
			v_p = rnn(qp_att,seq_len=)
		
		#--- 3. self-match attention ?
		with tf.variable_scope('self_matching'):
			self_att = dot_attention()
			rnn = gru()
			h_p = rnn(self,seq_len)

		#--- 4. output layer
		with tf.variable_scope('ouput'):
			with tf.variable_scope('pointer'):
				# how to calculate r^Q with pooling ???
				init_state = summ(u_q[])
				# bulid pointer
				pointer = ptr_net()
				logits1, logits2 = pointer(init_state,h_p,)

			with tf.variable_scope('predict'):
				outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              	  tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            	outer = tf.matrix_band_part(outer, 0, 15)
            	self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            	self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            	losses1 = tf.nn.softmax_cross_entropy_with_logits_V2()
            	losses2 = tf.nn.softmax_cross_entropy_with_logits_V2()
            	self.loss = tf.reduce_mean(losses1 + losses2)

		if is_train:
			self.lr = 
			self.opt = tf.train.AdadeltaOptimizer()
			grads = self.opt.compute_graientes(self.loss)
			gradientes, variables = zip(*grads)
			capped_grads,_ = tf.clip_by_global_norm(gradientes,)
			self.train_op = self.opt.apply_gradients(
				zip(capped_grads,variables),global_step=self.global_step)
