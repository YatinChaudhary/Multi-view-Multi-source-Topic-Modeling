import os
import argparse
import json
import numpy as np
import tensorflow as tf
import model.data as data
import model.model_MVT_MST as m
import model.evaluate as eval
import datetime
import json
import sys
import pickle

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from gensim.models.keyedvectors import KeyedVectors
import sklearn.metrics.pairwise as pw

#from fastText import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

home_dir = os.getenv("HOME")

dir(tf.contrib)

def loadGloveModel(vocab_docnade, gloveFile=None, params=None):
	if gloveFile is None:
		if params.hidden_size == 50:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.50d.txt")
		elif params.hidden_size == 100:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.100d.txt")
		elif params.hidden_size == 200:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.200d.txt")
		elif params.hidden_size == 300:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.300d.txt")
		else:
			print('Invalid dimension [%d] for Glove pretrained embedding matrix!!' %params.hidden_size)
			exit()

	print("Loading Glove Model")
	f = open(gloveFile, 'r')
	model = {}
	for line in f:
		splitLine = line.split()
		word = splitLine[0]
		embedding = np.array([float(val) for val in splitLine[1:]])
		model[word] = embedding
	print("Done.", len(model), " words loaded!")

	missing_words = 0
	docnade_glove_matrix = np.zeros((len(vocab_docnade), params.hidden_size), dtype=np.float32)
	for i, word in enumerate(vocab_docnade):
		if str(word).lower() in model.keys():
			if len(model[str(word).lower()]) == 0:
				docnade_glove_matrix[i, :] = np.zeros((params.hidden_size), dtype=np.float32)
				missing_words += 1
			else:
				docnade_glove_matrix[i, :] = np.array(model[str(word).lower()], dtype=np.float32)
		else:
			docnade_glove_matrix[i, :] = np.zeros((params.hidden_size), dtype=np.float32)
			missing_words += 1
	
	print("Total missing words:%d out of %d" %(missing_words, len(vocab_docnade)))
	#return model
	return docnade_glove_matrix

def loadBioModel(vocab_docnade, BioFile=None, params=None):
	print("Loading BioNLP Model")
	model = KeyedVectors.load_word2vec_format('/home/ubuntu/resources/pretrained_embeddings/PubMed-and-PMC-w2v.bin', binary=True)
	
	missing_words = 0
	docnade_glove_matrix = np.zeros((len(vocab_docnade), params.hidden_size), dtype=np.float32)
	for i, word in enumerate(vocab_docnade):
		try:
			docnade_glove_matrix[i, :] = model[str(word).lower()]
		except KeyError:
			missing_words += 1
			pass
	print("Total missing words:%d out of %d" %(missing_words, len(vocab_docnade)))

	#return model
	return docnade_glove_matrix

def loadFastTextModel(vocab_docnade, fasttext_file=None, params=None):
	print("Loading FastText Model")
	if fasttext_file is None:
		fasttext_file = '/home/ubuntu/resources/pretrained_embeddings/wiki.en.bin'
	model = load_model(fasttext_file)

	prior_vecs = []
	for word in vocab_docnade:
		prior_vecs.append(model.get_word_vector(word.strip()))
	docnade_fasttext_matrix = np.array(prior_vecs, dtype=np.float32)

	return docnade_fasttext_matrix

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

def train(model, dataset, params, vocab, docnade_bert_reps=None):
	log_dir = os.path.join(params.model, 'logs')
	model_dir_ir = os.path.join(params.model, 'model_ir')
	model_dir_ppl = os.path.join(params.model, 'model_ppl')

	with tf.Session(config=tf.ConfigProto(
		inter_op_parallelism_threads=params.num_cores,
		intra_op_parallelism_threads=params.num_cores,
		gpu_options=tf.GPUOptions(allow_growth=True)
	)) as session:
		avg_loss = tf.placeholder(tf.float32, [], 'loss_ph')
		tf.summary.scalar('loss', avg_loss)

		validation = tf.placeholder(tf.float32, [], 'validation_ph')
		validation_accuracy = tf.placeholder(tf.float32, [], 'validation_acc')
		tf.summary.scalar('validation', validation)
		tf.summary.scalar('validation_accuracy', validation_accuracy)

		summary_writer = tf.summary.FileWriter(log_dir, session.graph)
		summaries = tf.summary.merge_all()
		saver = tf.train.Saver(tf.global_variables())

		tf.local_variables_initializer().run()
		tf.global_variables_initializer().run()

		losses = []

		# This currently streams from disk. You set num_epochs=1 and
		# wrap this call with something like itertools.cycle to keep
		# this data in memory.
		# shuffle: the order of words in the sentence for DocNADE
		
		training_data = dataset.batches('training_docnade', params.batch_size, shuffle=True, multilabel=params.multi_label)

		best_val_IR = 0.0
		best_val_nll = np.inf
		best_val_ppl = np.inf
		best_val_disc_accuracy = 0.0

		best_test_IR = 0.0
		best_test_nll = np.inf
		best_test_ppl = np.inf
		best_test_disc_accuracy = 0.0
		
		patience = params.patience

		patience_count = 0
		patience_count_ir = 0
		best_train_nll = np.inf

		training_labels = np.array(
			[[row[0]] for index, row in dataset.rows('training_docnade', num_epochs=1)]
		)
		validation_labels = np.array(
			[[row[0]] for index, row in dataset.rows('validation_docnade', num_epochs=1)]
		)
		test_labels = np.array(
			[[row[0]] for index, row in dataset.rows('test_docnade', num_epochs=1)]
		)

		ppl_model = False
		ir_model = False
		
		for step in range(params.num_steps + 1):
			this_loss = -1.

			indices, y, x, seq_lengths = next(training_data)
			train_feed_dict = {
				model.x: x,
				model.y: y,
				model.seq_lengths: seq_lengths
			}

			if params.use_bert_prior:
				x_bert = get_bert_input(docnade_bert_reps["training_docnade"], indices, x.shape[1])
				train_feed_dict[model.x_bert] = x_bert
			
			_, loss, loss_unnormed = session.run([model.opt, model.loss_normed, model.loss_unnormed], feed_dict=train_feed_dict)
			this_loss = loss
			losses.append(this_loss)
			
			if (step % params.log_every == 0):
				print('{}: {:.6f}'.format(step, this_loss))

			if step and (step % params.validation_ppl_freq) == 0:
				ppl_model = True

				this_val_nll = []
				this_val_loss_normed = []
				# val_loss_unnormed is NLL
				this_val_nll_bw = []
				this_val_loss_normed_bw = []

				this_val_disc_accuracy = []
				
				for val_indices, val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
					val_feed_dict = {
						model.x: val_x,
						model.y: val_y,
						model.seq_lengths: val_seq_lengths
					}

					if params.use_bert_prior:
						val_x_bert = get_bert_input(docnade_bert_reps["validation_docnade"], val_indices, val_x.shape[1])
						val_feed_dict[model.x_bert] = val_x_bert
					
					val_loss_normed, val_loss_unnormed = session.run([model.loss_normed, model.loss_unnormed], feed_dict=val_feed_dict)

					this_val_nll.append(val_loss_unnormed)
					this_val_loss_normed.append(val_loss_normed)
				
				total_val_nll = np.mean(this_val_nll)
				total_val_ppl = np.exp(np.mean(this_val_loss_normed))

				if total_val_ppl < best_val_ppl:
					best_val_ppl = total_val_ppl
					print('saving: {}'.format(model_dir_ppl))
					saver.save(session, model_dir_ppl + '/model_ppl', global_step=1)

				# Early stopping
				if total_val_nll < best_val_nll:
					best_val_nll = total_val_nll
					patience_count = 0
				else:
					patience_count += 1

				print('This val PPL: {:.3f} (best val PPL: {:.3f},  best val loss: {:.3f}'.format(
					total_val_ppl,
					best_val_ppl or 0.0,
					best_val_nll
				))

				# logging information
				with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
					f.write("Step: %i,	val PPL: %s,	 best val PPL: %s,	best val loss: %s\n" % 
							(step, total_val_ppl, best_val_ppl, best_val_nll))

				if patience_count > patience:
					print("Early stopping criterion satisfied.")
					break
			
			if step and (step % params.validation_ir_freq) == 0:
				ir_model = True

				validation_vectors = m.vectors(
					model,
					dataset.batches(
						'validation_docnade',
						params.batch_size,
						num_epochs=1,
						shuffle=True,
						multilabel=params.multi_label
					),
					session,
					params,
					vocab=vocab,
					bert_reps=docnade_bert_reps["validation_docnade"]
				)

				training_vectors = m.vectors(
					model,
					dataset.batches(
						'training_docnade',
						params.batch_size,
						num_epochs=1,
						shuffle=True,
						multilabel=params.multi_label
					),
					session,
					params,
					vocab=vocab,
					bert_reps=docnade_bert_reps["training_docnade"]
				)
				
				val = eval.evaluate(
					training_vectors,
					validation_vectors,
					training_labels,
					validation_labels,
					recall=[0.02],
					num_classes=params.num_classes,
					multi_label=params.multi_label
				)[0]

				if val > best_val_IR:
					best_val_IR = val
					print('saving: {}'.format(model_dir_ir))
					saver.save(session, model_dir_ir + '/model_ir', global_step=1)
					patience_count_ir = 0
				else:
					patience_count_ir += 1
				
				print('This val IR: {:.3f} (best val IR: {:.3f})'.format(
					val,
					best_val_IR or 0.0
				))

				# logging information
				with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
					f.write("Step: %i,	val IR: %s,	best val IR: %s\n" % 
							(step, val, best_val_IR))
					
				if patience_count_ir > patience:
					print("Early stopping criterion satisfied.")
					break
			
		if ppl_model:
			saver.restore(session, tf.train.latest_checkpoint(model_dir_ppl))

			## validation set ppl

			this_val_nll = []
			this_val_loss_normed = []
			# val_loss_unnormed is NLL
			
			for val_indices, val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params.validation_bs, num_epochs=1, shuffle=False, multilabel=params.multi_label):
				val_feed_dict = {
					model.x: val_x,
					model.y: val_y,
					model.seq_lengths: val_seq_lengths
				}

				if params.use_bert_prior:
					val_x_bert = get_bert_input(docnade_bert_reps["validation_docnade"], val_indices, val_x.shape[1])
					val_feed_dict[model.x_bert] = val_x_bert
				
				val_loss_normed, val_loss_unnormed = session.run([model.loss_normed, model.loss_unnormed], feed_dict=val_feed_dict)

				this_val_nll.append(val_loss_unnormed)
				this_val_loss_normed.append(val_loss_normed)
			
			total_val_nll = np.mean(this_val_nll)
			total_val_ppl = np.exp(np.mean(this_val_loss_normed))

			## test set ppl

			this_test_nll = []
			this_test_loss_normed = []
			# test_loss_unnormed is NLL
			
			for test_indices, test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params.validation_bs, num_epochs=1, shuffle=False, multilabel=params.multi_label):
				test_feed_dict = {
					model.x: test_x,
					model.y: test_y,
					model.seq_lengths: test_seq_lengths
				}

				if params.use_bert_prior:
					test_x_bert = get_bert_input(docnade_bert_reps["test_docnade"], test_indices, test_x.shape[1])
					test_feed_dict[model.x_bert] = test_x_bert
				
				test_loss_normed, test_loss_unnormed = session.run([model.loss_normed, model.loss_unnormed], feed_dict=test_feed_dict)

				this_test_nll.append(test_loss_unnormed)
				this_test_loss_normed.append(test_loss_normed)
			
			total_test_nll = np.mean(this_test_nll)
			total_test_ppl = np.exp(np.mean(this_test_loss_normed))

			print('Val PPL: {:.3f},    Test PPL: {:.3f}'.format(
				total_val_ppl,
				total_test_ppl
			))

			# logging information
			with open(os.path.join(log_dir, "reload_info.txt"), "a") as f:
				f.write("\nVal PPL: %s,	 Test PPL: %s\n" % 
						(total_val_ppl, total_test_ppl))

			# Topics with W matrix
			W_target = session.run("embedding:0")
			
			top_n_topic_words = 20
			w_h_top_words_indices = []
			W_topics = W_target
			topics_list_W = []

			for h_num in range(np.array(W_topics).shape[1]):
				w_h_top_words_indices.append(np.argsort(W_topics[:, h_num])[::-1][:top_n_topic_words])

			with open(params.docnadeVocab, 'r') as f:
				vocab_docnade = [w.strip() for w in f.readlines()]

			with open(os.path.join(log_dir, "topics_ppl_W.txt"), "w") as f:
				for w_h_top_words_indx, h_num in zip(w_h_top_words_indices, range(len(w_h_top_words_indices))):
					w_h_top_words = [vocab_docnade[w_indx] for w_indx in w_h_top_words_indx]
					
					topics_list_W.append(w_h_top_words)
					
					print('h_num: %s' % h_num)
					print('w_h_top_words_indx: %s' % w_h_top_words_indx)
					print('w_h_top_words:%s' % w_h_top_words)
					print('----------------------------------------------------------------------')

					f.write('h_num: %s\n' % h_num)
					f.write('w_h_top_words_indx: %s\n' % w_h_top_words_indx)
					f.write('w_h_top_words:%s\n' % w_h_top_words)
					f.write('----------------------------------------------------------------------\n')

			# TOPIC COHERENCE

			top_n_word_in_each_topic_list = [5, 10, 15, 20]

			text_filenames = [
				params.trainfile,
				params.valfile,
				params.testfile
			]

			# read original text documents as list of words
			texts = []

			for file in text_filenames:
				print('filename:%s', file)
				for line in open(file, 'r').readlines():
					document = str(line).strip()
					texts.append(document.split())

			compute_coherence(texts, topics_list_W, top_n_word_in_each_topic_list, os.path.join(log_dir, "topics_coherence_W.txt"))

		if ir_model:
			saver.restore(session, tf.train.latest_checkpoint(model_dir_ir))

			ir_ratio_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
			#ir_ratio_list = [0.02]

			validation_vectors = m.vectors(
				model,
				dataset.batches(
					'validation_docnade',
					params.batch_size,
					num_epochs=1,
					shuffle=True,
					multilabel=params.multi_label
				),
				session,
				params,
				vocab=vocab,
				bert_reps=docnade_bert_reps["validation_docnade"]
			)

			test_vectors = m.vectors(
				model,
				dataset.batches(
					'test_docnade',
					params.batch_size,
					num_epochs=1,
					shuffle=True,
					multilabel=params.multi_label
				),
				session,
				params,
				vocab=vocab,
				bert_reps=docnade_bert_reps["test_docnade"]
			)

			training_vectors = m.vectors(
				model,
				dataset.batches(
					'training_docnade',
					params.batch_size,
					num_epochs=1,
					shuffle=True,
					multilabel=params.multi_label
				),
				session,
				params,
				vocab=vocab,
				bert_reps=docnade_bert_reps["training_docnade"]
			)

			val = eval.evaluate(
				training_vectors,
				validation_vectors,
				training_labels,
				validation_labels,
				recall=ir_ratio_list,
				num_classes=params.num_classes,
				multi_label=params.multi_label
			)

			test = eval.evaluate(
				training_vectors,
				test_vectors,
				training_labels,
				test_labels,
				recall=ir_ratio_list,
				num_classes=params.num_classes,
				multi_label=params.multi_label
			)

			print('This val IR: %s,    \nbest test IR: %s' % (val, test))

			# logging information
			with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
				f.write("IR Ratio: %s" % ir_ratio_list)
				f.write("\nval IR: %s,	\ntest IR: %s" % (val, test))

from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

def compute_coherence(texts, list_of_topics, top_n_word_in_each_topic_list, reload_model_dir):

	dictionary = Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]

	print('corpus len:%s' %len(corpus))
	print('dictionary:%s' %dictionary)
	# https://github.com/earthquakesan/palmetto-py
	# compute_topic_coherence: PMI and other coherence types
	# from palmettopy.palmetto import Palmetto
	# palmetto = Palmetto()

	# coherence_types = ["ca", "cp", "cv", "npmi", "uci", "umass"] # for palmetto library
	coherence_types = ["c_v"]#, 'u_mass', 'c_v', 'c_uci', 'c_npmi'] # ["c_v"] # 'u_mass', 'c_v', 'c_uci', 'c_npmi',
	avg_coh_scores_dict = {}

	best_coh_type_value_topci_indx = {}
	for top_n in top_n_word_in_each_topic_list:
		avg_coh_scores_dict[top_n]= []
		best_coh_type_value_topci_indx[top_n] = [0,  0, []] # score, topic_indx, topics words


	h_num = 0
	with open(reload_model_dir, "w") as f:
		for topic_words_all in list_of_topics:
			h_num += 1
			for top_n in top_n_word_in_each_topic_list:
				topic_words = [topic_words_all[:top_n]]
				for coh_type in coherence_types:
					try:
						print('top_n: %s Topic Num: %s \nTopic Words: %s' % (top_n, h_num, topic_words))
						f.write('top_n: %s Topic Num: %s \nTopic Words: %s\n' % (top_n, h_num, topic_words))
						# print('topic_words_top_10_abs[%s]:%s' % (h_num, topic_words_top_10_abs[h_num]))
						# PMI = palmetto.get_coherence(topic_words_top_10[h_num], coherence_type=coh_type)
						PMI = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence=coh_type, processes=2).get_coherence()

						avg_coh_scores_dict[top_n].append(PMI)

						if PMI > best_coh_type_value_topci_indx[top_n][0]:
							best_coh_type_value_topci_indx[top_n] = [PMI, top_n, topic_words]

						print('Coh_type:%s  Topic Num:%s COH score:%s' % (coh_type, h_num, PMI))
						f.write('Coh_type:%s  Topic Num:%s COH score:%s\n' % (coh_type, h_num, PMI))

						print('--------------------------------------------------------------')
					except:
						continue
				print('========================================================================================================')

		for top_n in top_n_word_in_each_topic_list:
			print('top scores for top_%s:%s' %(top_n, best_coh_type_value_topci_indx[top_n]))
			print('-------------------------------------------------------------------')
			f.write('top scores for top_%s:%s\n' %(top_n, best_coh_type_value_topci_indx[top_n]))
			f.write('-------------------------------------------------------------------\n')

		for top_n in top_n_word_in_each_topic_list:
			print('Avg COH for top_%s topic words: %s' %(top_n, np.mean(avg_coh_scores_dict[top_n])))
			print('-------------------------------------------------------------------')
			f.write('Avg COH for top_%s topic words: %s\n' %(top_n, np.mean(avg_coh_scores_dict[top_n])))
			f.write('-------------------------------------------------------------------\n')

def get_vectors_from_matrix(matrix, batches):
	# matrix: embedding matrix of shape = [vocab_size X embedding_size]
	vecs = []
	for _, x, seq_length in batches:
		temp_vec = np.zeros((matrix.shape[1]), dtype=np.float32)
		indices = x[0, :seq_length[0]]
		for index in indices:
			temp_vec += matrix[index, :]
		vecs.append(temp_vec)
	return np.array(vecs)

def softmax(X, theta = 1.0, axis = None):
	"""
	Compute the softmax of each element along an axis of X.

	Parameters
	----------
	X: ND-Array. Probably should be floats. 
	theta (optional): float parameter, used as a multiplier
		prior to exponentiation. Default = 1.0
	axis (optional): axis to compute values along. Default is the 
		first non-singleton axis.

	Returns an array the same size as X. The result will sum to 1
	along the specified axis.
	"""

	# make X at least 2d
	y = np.atleast_2d(X)

	# find axis
	if axis is None:
		axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

	# multiply y against the theta parameter, 
	y = y * float(theta)

	# subtract the max for numerical stability
	y = y - np.expand_dims(np.max(y, axis = axis), axis)
	
	# exponentiate y
	y = np.exp(y)

	# take the sum along the specified axis
	ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

	# finally: divide elementwise
	p = y / ax_sum

	# flatten if X was 1D
	if len(X.shape) == 1: p = p.flatten()

	return p


from math import *
from nltk.corpus import wordnet
def square_rooted(x):
	return round(sqrt(sum([a * a for a in x])), 3)

def cosine_similarity(x, y):
	numerator = sum(a * b for a, b in zip(x, y))
	denominator = square_rooted(x) * square_rooted(y)
	return round(numerator / float(denominator), 3)


def reload_evaluation_ir(params, training_vectors, test_vectors, W_matrix, suffix=""):

		### Information Retrieval

		dataset = data.Dataset(params['dataset'])
		#log_dir = os.path.join(params['model'], 'logs')
		log_dir = os.path.join("model", params['reload_model_dir'], 'logs')

		ir_ratio_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
		#ir_ratio_list = [0.02]

		training_labels = np.array(
			[[row[0]] for index, row in dataset.rows('training_docnade', num_epochs=1)]
		)

		test_labels = np.array(
			[[row[0]] for index, row in dataset.rows('test_docnade', num_epochs=1)]
		)

		import pdb; pdb.set_trace()
		test_ir_list = eval.evaluate(
			training_vectors,
			test_vectors,
			training_labels,
			test_labels,
			recall=ir_ratio_list,
			num_classes=params['num_classes'],
			multi_label=params['multi_label']
		)

		# logging information
		with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
			f.write("\n\nFractions list: %s" % (ir_ratio_list))
			f.write("\nTest IR: %s" % (test_ir_list))


def reload_evaluation_ppl(params, suffix=""):
	with tf.Session(config=tf.ConfigProto(
		inter_op_parallelism_threads=params['num_cores'],
		intra_op_parallelism_threads=params['num_cores'],
		gpu_options=tf.GPUOptions(allow_growth=True)
	)) as session_ppl:

		dataset = data.Dataset(params['dataset'])
		log_dir = os.path.join(params['model'], 'logs')
			
		saver_ppl = tf.train.import_meta_graph("./model/" + params['reload_model_dir'] + "/model_ppl/model_ppl-1.meta")
		saver_ppl.restore(session_ppl, tf.train.latest_checkpoint("./model/" + params['reload_model_dir'] + "/model_ppl/"))

		graph = tf.get_default_graph()

		x = graph.get_tensor_by_name("x:0")
		y = graph.get_tensor_by_name("y:0")
		seq_lengths = graph.get_tensor_by_name("seq_lengths:0")
		loss_normed = graph.get_tensor_by_name("loss_normed_x:0")
		loss_unnormed = graph.get_tensor_by_name("loss_unnormed_x:0")

		with open(params['dataset'] + "/vocab_docnade.vocab", "r") as f:
			vocab = [line.strip() for line in f.readlines()]

		# TODO: Validation PPL

		this_val_nll = []
		this_val_loss_normed = []
		# val_loss_unnormed is NLL
		this_val_nll_bw = []
		this_val_loss_normed_bw = []

		this_val_disc_accuracy = []
		
		for val_indices, val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params['validation_bs'], num_epochs=1, shuffle=False, multilabel=params['multi_label']):
			val_feed_dict = {
				x: val_x,
				y: val_y,
				seq_lengths: val_seq_lengths
			}
			
			val_loss_normed, val_loss_unnormed = session_ppl.run([loss_normed, loss_unnormed], feed_dict=val_feed_dict)

			this_val_nll.append(val_loss_unnormed)
			this_val_loss_normed.append(val_loss_normed)
		
		
		total_val_nll = np.mean(this_val_nll)
		total_val_ppl = np.exp(np.mean(this_val_loss_normed))

		print('Val PPL: {:.3f},	Val loss: {:.3f}\n'.format(
			total_val_ppl,
			total_val_nll
		))

		# logging information
		with open(os.path.join(log_dir, "reload_info_ppl_" + suffix + ".txt"), "w") as f:
			f.write("Val PPL: %s,	Val loss: %s" % 
					(total_val_ppl, total_val_nll))
		
		# TODO: Test PPL

		this_test_nll = []
		this_test_loss_normed = []
		this_test_nll_bw = []
		this_test_loss_normed_bw = []
		this_test_disc_accuracy = []
		
		for test_indices, test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params['test_bs'], num_epochs=1, shuffle=False, multilabel=params['multi_label']):
			test_feed_dict = {
				x: test_x,
				y: test_y,
				seq_lengths: test_seq_lengths
			}
			
			test_loss_normed, test_loss_unnormed = session_ppl.run([loss_normed, loss_unnormed], feed_dict=test_feed_dict)

			this_test_nll.append(test_loss_unnormed)
			this_test_loss_normed.append(test_loss_normed)
		
		total_test_nll = np.mean(this_test_nll)
		total_test_ppl = np.exp(np.mean(this_test_loss_normed))

		print('Test PPL: {:.3f},	Test loss: {:.3f}\n'.format(
			total_test_ppl,
			total_test_nll
		))

		# logging information
		with open(os.path.join(log_dir, "reload_info_ppl_" + suffix + ".txt"), "a") as f:
			f.write("\n\nTest PPL: %s,	Test loss: %s" % 
					(total_test_ppl, total_test_nll))

		W_target = session_ppl.run("embedding:0")
		
		top_n_words = 20

		# Nearest Neighbors
		with open(params['docnadeVocab'], 'r') as f:
			vocab_docnade = [w.strip() for w in f.readlines()]

		W = W_target
		
		sim_mat = pw.cosine_similarity(W, W)
		sim_mat[np.arange(len(vocab_docnade)), np.arange(len(vocab_docnade))] = -1.0

		sorted_indices = np.argsort(sim_mat, axis=1)[:, ::-1]
		
		with open(log_dir + "/nearest_neighbours.txt", "a") as f:
			for counter, indices in enumerate(sorted_indices[:, :top_n_words]):
				query_word = vocab_docnade[counter]
				nn_words = " | ".join([vocab_docnade[index] + " ( " + str(sim_mat[counter, index]) + " ) " for index in indices])
				line = query_word + " :: " + nn_words + "\n"
				f.write(line)

		
		bias_W_target = session_ppl.run("bias:0")
		U_target = session_ppl.run("U:0")
		bias_U_target = session_ppl.run("b:0")

		return W_target, bias_W_target, U_target, bias_U_target


def reload_evaluation_topics(W_target, U_target, params):

	log_dir = os.path.join(params['model'], 'logs')

	# Topics with W matrix

	top_n_topic_words = 20
	w_h_top_words_indices = []
	W_topics = W_target
	topics_list_W = []

	for h_num in range(np.array(W_topics).shape[1]):
		w_h_top_words_indices.append(np.argsort(W_topics[:, h_num])[::-1][:top_n_topic_words])

	with open(params['docnadeVocab'], 'r') as f:
		vocab_docnade = [w.strip() for w in f.readlines()]

	with open(os.path.join(log_dir, "topics_ppl_W.txt"), "w") as f:
		for w_h_top_words_indx, h_num in zip(w_h_top_words_indices, range(len(w_h_top_words_indices))):
			w_h_top_words = [vocab_docnade[w_indx] for w_indx in w_h_top_words_indx]
			
			topics_list_W.append(w_h_top_words)
			
			print('h_num: %s' % h_num)
			print('w_h_top_words_indx: %s' % w_h_top_words_indx)
			print('w_h_top_words:%s' % w_h_top_words)
			print('----------------------------------------------------------------------')

			f.write('h_num: %s\n' % h_num)
			f.write('w_h_top_words_indx: %s\n' % w_h_top_words_indx)
			f.write('w_h_top_words:%s\n' % w_h_top_words)
			f.write('----------------------------------------------------------------------\n')

	# TOPIC COHERENCE

	top_n_word_in_each_topic_list = [5, 10, 15, 20]

	text_filenames = [
		params['trainfile'],
		params['valfile'],
		params['testfile']
	]

	# read original text documents as list of words
	texts = []

	for file in text_filenames:
		print('filename:%s', file)
		for line in open(file, 'r').readlines():
			document = str(line).strip()
			texts.append(document.split())

	compute_coherence(texts, topics_list_W, top_n_word_in_each_topic_list, os.path.join(log_dir, "topics_coherence_W.txt"))


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def get_prior_matrix(prior_embedding_path, prior_vocab, docnade_vocab, hidden_size):
	prior_embedding_matrix = np.load(prior_embedding_path)
	
	W_old_indices = []
	W_new_indices = []
	W_old_matrix = []
	prior_matrix = np.zeros((len(docnade_vocab), hidden_size), dtype=np.float32)
	for i, word in enumerate(docnade_vocab):
		try:
			index = prior_vocab.index(word)
		except ValueError:
			continue
		prior_matrix[i, :] = prior_embedding_matrix[index, :]
		W_old_matrix.append(prior_embedding_matrix[index, :])
		W_old_indices.append(index)
		W_new_indices.append(i)
	
	return prior_matrix, np.array(W_old_matrix, dtype=np.float32), W_old_indices, W_new_indices

def main(args):
	args.reload = str2bool(args.reload)
	args.use_glove_prior = str2bool(args.use_glove_prior)
	args.use_bio_prior = str2bool(args.use_bio_prior)
	args.projection = str2bool(args.projection)
	args.concat_projection = str2bool(args.concat_projection)
	args.multi_label = str2bool(args.multi_label)
	args.gvt_loss = str2bool(args.gvt_loss)
	args.use_embeddings_prior = str2bool(args.use_embeddings_prior)
	args.use_fasttext_prior = str2bool(args.use_fasttext_prior)
	args.use_bert_prior = str2bool(args.use_bert_prior)

	if args.reload:
		with open("./model/" + args.reload_model_dir + "/params.json") as f:
			params = json.load(f)

		params['trainfile'] = args.trainfile
		params['valfile'] = args.valfile
		params['testfile'] = args.testfile

		params['reload_model_dir'] = args.reload_model_dir

		reload_ir = False
		if os.path.isdir("./model/" + args.reload_model_dir + "/model_ir"):
			reload_ir = True

		reload_ppl = False
		if os.path.isdir("./model/" + args.reload_model_dir + "/model_ppl"):
			reload_ppl = True

		# Reloading and evaluating on Perplexity, Topic Coherence and calculating Nearest Neighbors
		if reload_ppl:
			W_target, bias_W_target, U_target, bias_U_target = reload_evaluation_ppl(params, suffix="target")
			#reload_evaluation_topics(W_target, U_target, params)

		# Reloading and evaluating on Information Retrieval and Classification - F1
		if reload_ir:
			with tf.Session(config=tf.ConfigProto(
				inter_op_parallelism_threads=params['num_cores'],
				intra_op_parallelism_threads=params['num_cores'],
				gpu_options=tf.GPUOptions(allow_growth=True)
			)) as sess_ir:

				saver_ir = tf.train.import_meta_graph("./model/" + args.reload_model_dir + "/model_ir/model_ir-1.meta")
				saver_ir.restore(sess_ir, tf.train.latest_checkpoint("./model/" + args.reload_model_dir + "/model_ir/"))

				graph = tf.get_default_graph()

				x = graph.get_tensor_by_name("x:0")
				seq_lengths = graph.get_tensor_by_name("seq_lengths:0")
				last_hidden = graph.get_tensor_by_name("last_hidden:0")
				x_bert = graph.get_tensor_by_name("x_bert:0")

				dataset = data.Dataset(params['dataset'])
				#import pdb; pdb.set_trace()
				docnade_bert_reps = {
					"training_docnade": None, 
					"validation_docnade": None, 
					"test_docnade": None
				}
				if params["use_bert_prior"]:
					with open(os.path.join(params["bert_reps_path"], "training.pkl"), "rb") as f:
						docnade_bert_reps["training_docnade"] = pickle.load(f)
						bert_emb_shape = docnade_bert_reps["training_docnade"][0].shape[1]
					with open(os.path.join(params["bert_reps_path"], "validation.pkl"), "rb") as f:
						docnade_bert_reps["validation_docnade"] = pickle.load(f)
					with open(os.path.join(params["bert_reps_path"], "test.pkl"), "rb") as f:
						docnade_bert_reps["test_docnade"] = pickle.load(f)
					"""
					with open(os.path.join(params["bert_reps_path"][0], "training.pkl"), "rb") as f:
						docnade_bert_reps["training_docnade"] = pickle.load(f)
						bert_emb_shape = docnade_bert_reps["training_docnade"][0].shape[1]
					with open(os.path.join(params["bert_reps_path"][0], "validation.pkl"), "rb") as f:
						docnade_bert_reps["validation_docnade"] = pickle.load(f)
					with open(os.path.join(params["bert_reps_path"][0], "test.pkl"), "rb") as f:
						docnade_bert_reps["test_docnade"] = pickle.load(f)
					
					for reps_path in params["bert_reps_path"][1:]:
						with open(os.path.join(reps_path, "training.pkl"), "rb") as f:
							new_train_reps = pickle.load(f)
							bert_emb_shape += new_train_reps[0].shape[1]
						with open(os.path.join(reps_path, "validation.pkl"), "rb") as f:
							new_val_reps = pickle.load(f)
						with open(os.path.join(reps_path, "test.pkl"), "rb") as f:
							new_test_reps = pickle.load(f)
						
						for doc_num, doc_reps in new_train_reps.items():
							docnade_bert_reps["training_docnade"][doc_num] \
								= np.concatenate([docnade_bert_reps["training_docnade"][doc_num], doc_reps], axis=-1)
						for doc_num, doc_reps in new_val_reps.items():
							docnade_bert_reps["validation_docnade"][doc_num] \
								= np.concatenate([docnade_bert_reps["validation_docnade"][doc_num], doc_reps], axis=-1)
						for doc_num, doc_reps in new_test_reps.items():
							docnade_bert_reps["test_docnade"][doc_num] \
								= np.concatenate([docnade_bert_reps["test_docnade"][doc_num], doc_reps], axis=-1)
					"""
				hidden_vectors_tr = []
				for tr_indices, tr_y, tr_x, tr_seq_lengths in dataset.batches('training_docnade', batch_size=1, num_epochs=1, shuffle=True, multilabel=params['multi_label']):
					tr_feed_dict = {
						x: tr_x,
						seq_lengths: tr_seq_lengths
					}
					if params["use_bert_prior"]:
						tr_x_bert = get_bert_input(docnade_bert_reps["training_docnade"], tr_indices, tr_x.shape[1])
						tr_feed_dict[x_bert] = tr_x_bert

					hidden_vec = sess_ir.run([last_hidden], feed_dict=tr_feed_dict)
					hidden_vectors_tr.append(hidden_vec[0])
				hidden_vectors_tr = np.squeeze(np.array(hidden_vectors_tr, dtype=np.float32))
				
				hidden_vectors_test = []
				for te_indices, te_y, te_x, te_seq_lengths in dataset.batches('test_docnade', batch_size=1, num_epochs=1, shuffle=True, multilabel=params['multi_label']):
					te_feed_dict = {
						x: te_x,
						seq_lengths: te_seq_lengths
					}
					if params["use_bert_prior"]:
						tr_x_bert = get_bert_input(docnade_bert_reps["test_docnade"], te_indices, te_x.shape[1])
						te_feed_dict[x_bert] = tr_x_bert
					
					hidden_vec = sess_ir.run([last_hidden], feed_dict=te_feed_dict)
					hidden_vectors_test.append(hidden_vec[0])
				hidden_vectors_test = np.squeeze(np.array(hidden_vectors_test, dtype=np.float32))

				W_target = sess_ir.run("embedding:0") 
			
			reload_evaluation_ir(params, hidden_vectors_tr, hidden_vectors_test, 
								W_target, suffix="target")
	else:

		x = tf.placeholder(tf.int32, shape=(None, None), name='x')
		x_bw = tf.placeholder(tf.int32, shape=(None, None), name='x_bw')
		if args.multi_label:
			y = tf.placeholder(tf.string, shape=(None), name='y')
		else:
			y = tf.placeholder(tf.int32, shape=(None), name='y')
		seq_lengths = tf.placeholder(tf.int32, shape=(None), name='seq_lengths')

		if args.use_bert_prior:
			x_bert = tf.placeholder(tf.float32, shape=(None, None, None), name='x_bert')
		else:
			x_bert = None

		now = datetime.datetime.now()

		if args.use_glove_prior:
			args.model += "_emb_glove_" + str(args.lambda_glove)

		if args.use_bio_prior:
			args.model += "_emb_bio_" + str(args.lambda_glove)

		if args.use_embeddings_prior:
			args.model += "_emb_lambda_" + str(args.lambda_embeddings) + "_" + "_".join([str(lamb) for lamb in args.lambda_embeddings_list])
		
		if args.use_fasttext_prior:
			args.model += "_ftt_"

		if args.use_bert_prior:
			args.model += "_bert_"
		
		if args.W_pretrained_path or args.U_pretrained_path:
			args.model += "_pretr_reload_"
		
		args.model +=  "_act_" + str(args.activation) + "_hid_" + str(args.hidden_size) \
						+ "_vocab_" + str(args.vocab_size) + "_lr_" + str(args.learning_rate) \

		if args.gvt_loss:
			args.model += "_gvt_loss_" + str(args.gvt_loss) + "_" + str(args.gvt_lambda) + "_".join([str(lamb) for lamb in args.gvt_lambda_init])

		if args.projection:
			args.model += "_projection"

		if args.concat_projection:
			args.model += "_cp_" + str(args.concat_projection_lambda) + "_"
		
		args.model += "_" + str(now.day) + "_" + str(now.month) + "_" + str(now.year)
		
		if not os.path.isdir(args.model):
			os.mkdir(args.model)

		docnade_vocab = args.docnadeVocab
		with open(docnade_vocab, 'r') as f:
			vocab_docnade = [w.strip() for w in f.readlines()]

		with open(os.path.join(args.model, 'params.json'), 'w') as f:
			f.write(json.dumps(vars(args)))

		dataset = data.Dataset(args.dataset)

		docnade_glove_matrix = None
		if args.use_glove_prior:
			if args.use_bio_prior:
				#docnade_glove_matrix = np.load(args.dataset + "/glove_vectors_bio.npy")
				docnade_glove_matrix = np.load(args.dataset + "/word2vec_vectors_bio.npy")
			else:
				docnade_glove_matrix = np.load(args.dataset + "/glove_vectors.npy")

		docnade_fasttext_matrix = None
		if args.use_fasttext_prior:
			if args.use_bio_prior:
				docnade_fasttext_matrix = np.load(args.dataset + "/fasttext_vectors_bio.npy")
			else:
				docnade_fasttext_matrix = np.load(args.dataset + "/fasttext_vectors.npy")	

		docnade_bert_reps = {
			"training_docnade": None, 
			"validation_docnade": None, 
			"test_docnade": None
		}
		bert_emb_shape = 0
		if args.use_bert_prior:
			with open(os.path.join(args.bert_reps_path[0], "training.pkl"), "rb") as f:
				docnade_bert_reps["training_docnade"] = pickle.load(f)
				bert_emb_shape = docnade_bert_reps["training_docnade"][0].shape[1]
			with open(os.path.join(args.bert_reps_path[0], "validation.pkl"), "rb") as f:
				docnade_bert_reps["validation_docnade"] = pickle.load(f)
			with open(os.path.join(args.bert_reps_path[0], "test.pkl"), "rb") as f:
				docnade_bert_reps["test_docnade"] = pickle.load(f)
			
			for reps_path in args.bert_reps_path[1:]:
				with open(os.path.join(reps_path, "training.pkl"), "rb") as f:
					new_train_reps = pickle.load(f)
					bert_emb_shape += new_train_reps[0].shape[1]
				with open(os.path.join(reps_path, "validation.pkl"), "rb") as f:
					new_val_reps = pickle.load(f)
				with open(os.path.join(reps_path, "test.pkl"), "rb") as f:
					new_test_reps = pickle.load(f)
				
				for doc_num, doc_reps in new_train_reps.items():
					docnade_bert_reps["training_docnade"][doc_num] \
						= np.concatenate([docnade_bert_reps["training_docnade"][doc_num], doc_reps], axis=-1)
				for doc_num, doc_reps in new_val_reps.items():
					docnade_bert_reps["validation_docnade"][doc_num] \
						= np.concatenate([docnade_bert_reps["validation_docnade"][doc_num], doc_reps], axis=-1)
				for doc_num, doc_reps in new_test_reps.items():
					docnade_bert_reps["test_docnade"][doc_num] \
						= np.concatenate([docnade_bert_reps["test_docnade"][doc_num], doc_reps], axis=-1)
		
		W_pretrained_matrix = None
		if args.W_pretrained_path:
			W_pretrained_matrix = np.load(args.W_pretrained_path)
			print("pretrained W loaded.")

		U_pretrained_matrix = None
		if args.U_pretrained_path:
			U_pretrained_matrix = np.load(args.U_pretrained_path)
			print("pretrained U loaded.")

		W_old_indices_list = []
		W_new_indices_list = []
		W_old_matrices_list = []
		W_embeddings_matrices_list = []
		if args.use_embeddings_prior or args.gvt_loss:
			for i, W_old_path in enumerate(args.W_old_path_list):
				with open(args.W_old_vocab_path_list[i], "r") as f:
					temp_vocab = [str(word).lower().strip() for word in f.readlines()]

				prior_matrix, W_old_matrix, W_old_indices, W_new_indices = get_prior_matrix(W_old_path, temp_vocab, vocab_docnade, args.hidden_size)
				W_embeddings_matrices_list.append(prior_matrix)
				W_old_matrices_list.append(W_old_matrix)
				W_old_indices_list.append(W_old_indices)
				W_new_indices_list.append(W_new_indices)
			print("Loaded W_embeddings_matrices_list and W_embeddings_indices_list.")

			args.lambda_embeddings_list = np.array(args.lambda_embeddings_list, dtype=np.float32)
		
		if args.gvt_loss:
			args.gvt_lambda_init = np.array(args.gvt_lambda_init, dtype=np.float32)
		
		model = m.DocNADE_MVT_MST(x, y, seq_lengths, args, \
							W_old_list=W_old_matrices_list, \
							W_embeddings_matrices_list=W_embeddings_matrices_list, W_old_indices_list=W_old_indices_list, \
							lambda_embeddings_list=args.lambda_embeddings_list, W_new_indices_list=W_new_indices_list, \
							W_pretrained=W_pretrained_matrix, U_pretrained=U_pretrained_matrix, \
							glove_embeddings=docnade_glove_matrix, lambda_glove=args.lambda_glove, \
							fasttext_embeddings=docnade_fasttext_matrix, 
							x_bert=x_bert, bert_emb_shape=bert_emb_shape)
		print("DocNADE created")
		
		train(model, dataset, args, vocab_docnade, docnade_bert_reps=docnade_bert_reps)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, required=True,
						help='path to model output directory')
	parser.add_argument('--dataset', type=str, required=True,
						help='path to the input dataset')
	parser.add_argument('--vocab-size', type=int, default=2000,
						help='the vocab size')
	parser.add_argument('--hidden-size', type=int, default=50,
						help='size of the hidden layer')
	parser.add_argument('--activation', type=str, default='tanh',
						help='which activation to use: sigmoid|tanh')
	parser.add_argument('--learning-rate', type=float, default=0.0004,
						help='initial learning rate')
	parser.add_argument('--num-steps', type=int, default=50000,
						help='the number of steps to train for')
	parser.add_argument('--batch-size', type=int, default=64,
						help='the batch size')
	parser.add_argument('--num-cores', type=int, default=2,
						help='the number of CPU cores to use')
	parser.add_argument('--log-every', type=int, default=10,
						help='print loss after this many steps')
	parser.add_argument('--validation-ppl-freq', type=int, default=500,
						help='print loss after this many steps')
	parser.add_argument('--num-classes', type=int, default=-1,
						help='number of classes')
	parser.add_argument('--use-glove-prior', type=str, default="False",
						help='whether to use GloVe embeddings as prior information')
	parser.add_argument('--use-bio-prior', type=str, default="False",
						help='whether to use BioNLP embeddings as prior information')
	parser.add_argument('--docnadeVocab', type=str, default="False",
						help='path to vocabulary file used by DocNADE')
	parser.add_argument('--test-ppl-freq', type=int, default=100,
						help='print and log test PPL after this many steps')
	parser.add_argument('--test-ir-freq', type=int, default=100,
						help='print and log test IR after this many steps')
	parser.add_argument('--patience', type=int, default=10,
						help='print and log test IR after this many steps')
	parser.add_argument('--validation-bs', type=int, default=64,
						help='the batch size for validation evaluation')
	parser.add_argument('--test-bs', type=int, default=64,
						help='the batch size for test evaluation')
	parser.add_argument('--validation-ir-freq', type=int, default=500,
						help='print loss after this many steps')
	parser.add_argument('--projection', type=str, default="False",
						help='whether to project prior embeddings or not')
	parser.add_argument('--reload', type=str, default="False",
						help='whether to reload model or not')
	parser.add_argument('--reload-model-dir', type=str,
						help='path for model to be reloaded')
	parser.add_argument('--multi-label', type=str, default="False",
						help='whether dataset is multi-label or not')
	parser.add_argument('--trainfile', type=str, default="",
						help='path to train text file')
	parser.add_argument('--valfile', type=str, default="",
						help='path to validation text file')
	parser.add_argument('--testfile', type=str, default="",
						help='path to test text file')
	parser.add_argument('--lambda-glove', type=float, default=0.0,
						help='combination weight for prior GloVe embeddings into docnade')
	parser.add_argument('--W-pretrained-path', type=str, default="",
						help='path for pretrained W matrix')
	parser.add_argument('--U-pretrained-path', type=str, default="",
						help='path for pretrained U matrix')
	parser.add_argument('--gvt-loss', type=str, default="False",
						help='whether to include LL loss')
	parser.add_argument('--gvt-lambda', type=str, default="automatic",
						help='"automatic" or "manual"')
	parser.add_argument('--gvt-lambda-init', type=float, nargs='+', default=[],
						help='"automatic" or "manual"')
	parser.add_argument('--W-old-path-list', type=str, nargs='+', default=[],
						help='path to the W matrices of source datasets')
	parser.add_argument('--U-old-path-list', type=str, nargs='+', default=[],
						help='path to the U matrices of source datasets')
	parser.add_argument('--W-old-vocab-path-list', type=str, nargs='+', default=[],
						help='path to the vocab of source datasets')
	parser.add_argument('--use-embeddings-prior', type=str, default="False",
						help='whether to embedings as prior or not')
	parser.add_argument('--lambda-embeddings', type=str, default="",
						help='make embeddings lambda trainable or not')
	parser.add_argument('--lambda-embeddings-list', type=float, nargs='+', default=[],
						help='list of lambda for every embedding prior')
	parser.add_argument('--use-fasttext-prior', type=str, default="False",
						help='whether to use fasttext embedings as prior or not')
	parser.add_argument('--concat-projection', type=str, default="False",
						help='whether to use projection with concatenation of prior embeddings')
	parser.add_argument('--concat-projection-lambda', type=float, default=1.0,
						help='value of lambda for concat projection')

	parser.add_argument('--use-bert-prior', type=str, default="False",
						help='whether to bert contextualized embedings as prior or not')
	parser.add_argument('--bert-reps-path', type=str, nargs='+', default=[],
						help='path to bert contextualized embedings as prior or not')

	return parser.parse_args()


if __name__ == '__main__':
	main(parse_args())
