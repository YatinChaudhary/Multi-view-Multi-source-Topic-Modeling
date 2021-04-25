import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

seed = 42
tf_op_seed = 1234

np.random.seed(seed)
tf.set_random_seed(seed)

def get_bert_input(reps, indices, max_length):
	bert_inputs = []
	for index in indices:
		inputs = reps[index]
		if inputs.shape[0] < max_length:
			inputs = np.concatenate([inputs, np.zeros((max_length - inputs.shape[0], inputs.shape[1]), dtype=np.float32)], axis=0)
			assert(inputs.shape[0] == max_length)
		if inputs.shape[0] > max_length:
			raise ValueError("Dimension mismatch.")
		bert_inputs.append(inputs)
	bert_inputs = np.stack(bert_inputs, axis=0)
	assert(bert_inputs.shape[0] == len(indices))
	return bert_inputs

def vectors(model, data, session, params, vocab=None, bert_reps=None):
	vecs = []
	for indices, _, x, seq_lengths in data:
		feed_dict = {
			model.x: x,
			model.seq_lengths: seq_lengths
		}

		if params.use_bert_prior:
			x_bert = get_bert_input(bert_reps, indices, x.shape[1])
			feed_dict[model.x_bert] = x_bert
		vecs.extend(
			session.run([model.h], feed_dict=feed_dict)[0]
		)
	return np.array(vecs)

def loss(model, data, session):
	loss = []
	for _, x, seq_lengths in data:
		loss.append(
			session.run([model.loss], feed_dict={
				model.x: x,
				model.seq_lengths: seq_lengths
			})[0]
		)
	return sum(loss) / len(loss)

def gradients(opt, loss, vars, step, max_gradient_norm=None, dont_clip=[]):
	gradients = opt.compute_gradients(loss, vars)
	if max_gradient_norm is not None:
		to_clip = [(g, v) for g, v in gradients if v.name not in dont_clip]
		not_clipped = [(g, v) for g, v in gradients if v.name in dont_clip]
		gradients, variables = zip(*to_clip)
		clipped_gradients, _ = clip_ops.clip_by_global_norm(
			gradients,
			max_gradient_norm
		)
		gradients = list(zip(clipped_gradients, variables)) + not_clipped

	# Add histograms for variables, gradients and gradient norms
	for gradient, variable in gradients:
		if isinstance(gradient, ops.IndexedSlices):
			grad_values = gradient.values
		else:
			grad_values = gradient
		if grad_values is None:
			print('warning: missing gradient: {}'.format(variable.name))
		if grad_values is not None:
			tf.summary.histogram(variable.name, variable)
			tf.summary.histogram(variable.name + '/gradients', grad_values)
			tf.summary.histogram(
				variable.name + '/gradient_norm',
				clip_ops.global_norm([grad_values])
			)

	return opt.apply_gradients(gradients, global_step=step)

def linear_TL(input, output_dim, scope=None, stddev=None, U_pretrained=None):
	const = tf.constant_initializer(0.0)

	if U_pretrained is None:
		if stddev:
			norm = tf.random_normal_initializer(stddev=stddev, seed=tf_op_seed)
		else:
			norm = tf.random_normal_initializer(
				stddev=np.sqrt(2.0 / input.get_shape()[1].value),
				seed=tf_op_seed
			)

		U = tf.get_variable(
			'U',
			[input.get_shape()[1], output_dim],
			initializer=norm
		)
	else:
		U = tf.get_variable(
			'U',
			initializer=U_pretrained
		)

	b = tf.get_variable('b', [output_dim], initializer=const)

	input_logits = tf.nn.xw_plus_b(input, U, b)
	
	return input_logits, U

def masked_sequence_cross_entropy_loss(
	x,
	seq_lengths,
	logits,
	loss_function=None,
	norm_by_seq_lengths=True,
	name=""
):
	'''
	Compute the cross-entropy loss between all elements in x and logits.
	Masks out the loss for all positions greater than the sequence
	length (as we expect that sequences may be padded).

	Optionally, also either use a different loss function (eg: sampled
	softmax), and/or normalise the loss for each sequence by the
	sequence length.
	'''
	batch_size = tf.shape(x)[0]
	labels = tf.reshape(x, [-1])

	
	max_doc_length = tf.shape(x)[1]
	mask = tf.less(
		tf.range(0, max_doc_length, 1),
		tf.reshape(seq_lengths, [batch_size, 1])
	)
	mask = tf.reshape(mask, [-1])
	mask = tf.to_float(tf.where(
		mask,
		tf.ones_like(labels, dtype=tf.float32),
		tf.zeros_like(labels, dtype=tf.float32)
	))

	if loss_function is None:
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=logits,
			labels=labels
		)
	else:
		loss = loss_function(logits, labels)
	loss *= mask
	loss = tf.reshape(loss, [batch_size, -1])
	loss = tf.reduce_sum(loss, axis=1)
	loss_unnormed = loss
	if norm_by_seq_lengths:
		loss = loss / tf.to_float(seq_lengths)
	return tf.reduce_mean(loss, name="loss_normed_" + name), labels, mask, tf.reduce_mean(loss_unnormed, name="loss_unnormed_" + name)

class DocNADE_MVT_MST(object):
	def __init__(self, x, y, seq_lengths, params, W_old_list=None, 
				 W_embeddings_matrices_list=None, W_old_indices_list=None, 
				 lambda_embeddings_list=None, W_new_indices_list=None,
				 W_pretrained=None, U_pretrained=None, 
				 glove_embeddings=None, lambda_glove=0.0,
				 fasttext_embeddings=None, x_bert=None, bert_emb_shape=0):
		self.x = x
		self.y = y
		self.seq_lengths = seq_lengths
		if params.use_bert_prior:
			self.x_bert = x_bert

		batch_size = tf.shape(x)[0]
		batch_length = tf.shape(x)[1]
		self.b_s = tf.shape(x)
		self.W_old_list = W_old_list
		self.lambda_embeddings_list = lambda_embeddings_list
		self.lambda_glove = lambda_glove

		# Do an embedding lookup for each word in each sequence
		with tf.device('/cpu:0'):
			if W_pretrained is None:
				max_embed_init = 1.0 / (params.vocab_size * params.hidden_size)
				W = tf.get_variable(
					'embedding',
					[params.vocab_size, params.hidden_size],
					initializer=tf.random_uniform_initializer(
						maxval=max_embed_init,
						seed=tf_op_seed
					)
				)
			else:
				W = tf.get_variable(
					'embedding',
					initializer=W_pretrained
				)
			self.embeddings = tf.nn.embedding_lookup(W, x, name='embeddings')
			
			projection_input_dim = 0
			self.prior_embeddings = []
			if not glove_embeddings is None:
				glove_prior = tf.get_variable(
					'glove_prior',
					initializer=glove_embeddings,
					trainable=False
				)
				self.embeddings_prior_glove = tf.nn.embedding_lookup(glove_prior, x, name='glove_embeddings')
				
				if params.concat_projection:
					self.prior_embeddings.append(self.embeddings_prior_glove)
					projection_input_dim += glove_embeddings.shape[1]
				else:
					# Lambda multiplication
					if not self.lambda_glove < 0.0:
						self.embeddings_prior_glove = tf.scalar_mul(self.lambda_glove, self.embeddings_prior_glove)
					self.embeddings = tf.add(self.embeddings, self.embeddings_prior_glove)
			
			if not fasttext_embeddings is None:
				fasttext_prior = tf.get_variable(
					'fasttext_prior',
					initializer=fasttext_embeddings,
					trainable=False
				)
				self.embeddings_prior_fasttext = tf.nn.embedding_lookup(fasttext_prior, x, name='fasttext_embeddings')

				if params.concat_projection:
					self.prior_embeddings.append(self.embeddings_prior_fasttext)
					projection_input_dim += fasttext_embeddings.shape[1]
				elif fasttext_embeddings.shape[1] != params.hidden_size:
					pass
				else:
					self.embeddings = tf.add(self.embeddings, self.embeddings_prior_fasttext)
			
			if params.use_bert_prior:
				if params.concat_projection:
					self.prior_embeddings.append(self.x_bert)
					projection_input_dim += bert_emb_shape
			
			bias = tf.get_variable(
				'bias',
				[params.hidden_size],
				initializer=tf.constant_initializer(0)
			)

			if params.use_embeddings_prior:
				if not params.concat_projection:
					if params.lambda_embeddings == "manual":
						self.embeddings_lambda_list = tf.get_variable(
							'embeddings_lambda_list',
							initializer=lambda_embeddings_list,
							trainable=False
						)
					elif params.lambda_embeddings == "automatic":
						embeddings_lambda_list = tf.get_variable(
							'embeddings_lambda_list_unclipped',
							initializer=lambda_embeddings_list,
							trainable=True
						)
						self.embeddings_lambda_list = tf.clip_by_value(embeddings_lambda_list, 0.0, 1.0, name='embeddings_lambda_list')
					else:
						print("Invalid parameter value for lambda_embeddings: ", params.lambda_embeddings)
						sys.exit()

				self.W_prior_list = []
				self.embeddings_prior_list = []
				for i, W_embeddings in enumerate(W_embeddings_matrices_list):
					W_prior = tf.get_variable(
						'embedding_prior_' + str(i),
						initializer=W_embeddings,
						trainable=False
					)
					embedding_prior = tf.nn.embedding_lookup(W_prior, x)

					if params.concat_projection:
						self.prior_embeddings.append(embedding_prior)
						projection_input_dim += W_embeddings.shape[1]
					else:
						embedding_prior_with_lambda = tf.scalar_mul(self.embeddings_lambda_list[i], embedding_prior)
						self.embeddings = tf.add(self.embeddings, embedding_prior_with_lambda)

					self.W_prior_list.append(W_prior)
					self.embeddings_prior_list.append(embedding_prior)
			
			if params.concat_projection and (projection_input_dim != 0):
				print("Projection.")
				max_embed_init_projection = 1.0 / (projection_input_dim * params.hidden_size)
				prior_emb_projection = tf.get_variable(
					'prior_emb_projection',
					[projection_input_dim, params.hidden_size],
					initializer=tf.random_uniform_initializer(
						maxval=max_embed_init_projection,
						seed=tf_op_seed
					)
				)
				combined_embedding_prior = tf.concat(self.prior_embeddings, axis=2, name='prior_embeddings_concat')
				combined_embedding_prior = tf.reshape(
					combined_embedding_prior, 
					[-1, projection_input_dim],
					name='embeddings_prior_fasttext_bert_reshape_before_proj'
				)
				
				combined_embedding_prior = tf.matmul(
					combined_embedding_prior,
					prior_emb_projection,
					name='embeddings_prior_fasttext_bert_projected'
				)
				
				combined_embedding_prior = tf.reshape(
					combined_embedding_prior, 
					[batch_size, batch_length, params.hidden_size],
					name='embeddings_prior_fasttext_bert_reshape_after_proj'
				)
				
				combined_embedding_prior = tf.scalar_mul(params.concat_projection_lambda, combined_embedding_prior)
				self.embeddings = tf.add(self.embeddings, combined_embedding_prior, name='embeddings_combined')
			
			if params.gvt_loss:
				if params.gvt_lambda == "manual":
					self.gvt_lambda_list = tf.get_variable(
						'gvt_lambda_list',
						initializer=params.gvt_lambda_init,
						trainable=False
					)
				elif params.gvt_lambda == "automatic":
					gvt_lambda_list = tf.get_variable(
						'gvt_lambda_list_unclipped',
						initializer=params.gvt_lambda_init,
						trainable=True
					)
					self.gvt_lambda_list = tf.clip_by_value(gvt_lambda_list, 0.0, 1.0, name='gvt_lambda_list')
				else:
					print("Invalid parameter value for sal_gamma: ", params.sal_gamma)
					sys.exit()

				if params.projection:
					self.gvt_proj_matrices_W = []
					self.gvt_proj_matrices_U = []

					max_embed_init = 1.0 / (params.hidden_size * params.hidden_size)

					for i, gvt_temp_init in enumerate(params.gvt_lambda_init):
						gvt_proj_temp_W = tf.get_variable(
							'gvt_projection_W_' + str(i),
							[params.hidden_size, params.hidden_size],
							initializer=tf.random_uniform_initializer(
								maxval=max_embed_init,
								seed=tf_op_seed
							)
						)
						self.gvt_proj_matrices_W.append(gvt_proj_temp_W)

		# Compute the hidden layer inputs: each gets summed embeddings of
		# previous words
		def sum_embeddings(previous, current):
			return previous + current

		h = tf.scan(sum_embeddings, tf.transpose(self.embeddings, [1, 2, 0]))
		h = tf.transpose(h, [2, 0, 1])
		
		h = tf.concat([
			tf.zeros([batch_size, 1, params.hidden_size], dtype=tf.float32), h
		], axis=1)
		
		self.pre_act = h

		# Apply activation
		if params.activation == 'sigmoid':
			h = tf.sigmoid(h + bias)
		elif params.activation == 'tanh':
			h = tf.tanh(h + bias)
		elif params.activation == 'relu':
			h = tf.nn.relu(h + bias)
		else:
			print('Invalid value for activation: %s' % (params.activation))
			exit()
		
		self.aft_act = h

		# Extract final state for each sequence in the batch
		indices = tf.stack([
			tf.range(batch_size),
			tf.to_int32(seq_lengths)
		], axis=1)
		self.indices = indices
		self.h = tf.gather_nd(h, indices, name='last_hidden')

		h = h[:, :-1, :]
		h = tf.reshape(h, [-1, params.hidden_size])
		
		self.logits, U_new = linear_TL(h, params.vocab_size, scope='softmax', U_pretrained=U_pretrained)
		loss_function = None
		
		self.loss_normed, self.labels, self.mask, self.loss_unnormed = masked_sequence_cross_entropy_loss(
			x,
			seq_lengths,
			self.logits,
			loss_function=loss_function,
			norm_by_seq_lengths=True,
			name="x"
		)

		self.total_loss = tf.identity(self.loss_unnormed, name="total_loss")

		if params.gvt_loss:
			W_reg_loss = 0.0
			for i, (W_old, W_new_indices) in enumerate(zip(W_old_list, W_new_indices_list)):
				W_new_temp = tf.gather(W, W_new_indices, axis=0)
				if params.projection:
					W_new_temp_proj = tf.matmul(W_new_temp, self.gvt_proj_matrices_W[i])
				else:
					W_new_temp_proj = W_new_temp
				W_l2_loss = tf.nn.l2_loss((W_new_temp_proj - W_old), name='l2_loss_W_old_' + str(i))
				W_reg_loss += self.gvt_lambda_list[i] * W_l2_loss
			
			self.total_reg_loss = W_reg_loss
			self.total_loss += self.total_reg_loss
			self.W_new_temp = W_new_temp

		# Optimiser
		step = tf.Variable(0, trainable=False)
		self.opt = gradients(
			opt=tf.train.AdamOptimizer(learning_rate=params.learning_rate),
			loss=self.total_loss,
			vars=tf.trainable_variables(),
			step=step
		)