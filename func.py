import tensorflow as tf



def dot_attention(inputs,memory,mask,hidden,keep_probs=1.0,is_train=None,
	scope='dot_attention'):
	with tf.varivable_scope(scope):
		d_inputs = dropout(inputs,keep_prob=keep_probs,)
		d_memory = dropout()
		JX = tf.shape(inputs)[1]
		with tf.variable_scope('attention'):
			# HKUST use the add attention but R-Net paper use the add attention
			inputs_dense = tf.nn.relu(dense())
			memory_dense = tf.nn
			s_j = tf.matmul(inputs_dense,tf.transpose(memory_dense))
			mask = tf.tile(tf.expand_dims(mask,axis=1),[1,JX,1])
			# why using mask
			a_t = tf.nn.softmax(softmax_mask(s_j,mask))
			# c_t = a_t * u_q where the u_q = memory
			c_t = tf.matmul(logits,memory)
			qp_attention = tf.concat([inputs,c_t],axis=2)  

		with tf.variable_scope('gate');

			g_t = tf.nn.sigmoid(dense(qp_attention,dim))
			return g_t*qp_attention
			
def dense(inputs,hidden,use_bias=True,scope='dense'):
	with tf.variable_scope(scope):
		shape = tf.shape(inputs)
		dim = inputs.get_shape().as_list()[-1]
		output_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()-1))
				       ] + [hidden]
		flat_inputs = tf.reshape(inputs,[-1,dim])
		W = tf.get_variable('W',[dim,hidden])
		res = tf.matmul(flat_inputs,W)
		if use_bias:
			b = tf.get_variable('b',[hidden],initializer=tf.constant_initializer(0.))
			res = tf.nn.bias_add(res,b)
		res = tf.reshape(res,output_shape)
		return res