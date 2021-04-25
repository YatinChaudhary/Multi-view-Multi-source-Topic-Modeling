import numpy as np
import sklearn.metrics.pairwise as pw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import csv


def compare_labels(train_labels, test_label, label_type="", evaluation_type="", labels_to_count=[]):

	vec_goodLabel = []

	train_labels = np.asarray(train_labels)

	if label_type == "single":
		if not (isinstance(test_label, int) or isinstance(train_labels[0], int)):
			print("Labels are not instances of int")
			exit()

		test_labels = np.ones(train_labels.shape[0], dtype=np.float32) * test_label

		vec_goodLabel = np.array((train_labels == test_labels), dtype=np.int8)
	elif label_type == "multi":
		if not len(train_labels[0]) == len(test_label):
			print("Mismatched label vector length")
			exit()

		test_labels = np.asarray(test_label)
		labels_comparison_vec = np.dot(train_labels, test_labels)

		if evaluation_type == "relaxed":
			vec_goodLabel = np.array((labels_comparison_vec != 0), dtype=np.int8)

		elif evaluation_type == "strict":
			test_label_vec = np.ones(train_labels.shape[0]) * np.sum(test_label)
			vec_goodLabel = np.array((labels_comparison_vec == test_label_vec), dtype=np.int8)

		else:
			print("Invalid evaluation_type value.")

	else:
		print("Invalid label_type value.")

	return vec_goodLabel

def perform_IR_prec(kernel_matrix_test, train_labels, test_labels, list_percRetrieval=None, single_precision=False, label_type="", evaluation="", index2label_dict=None, labels_to_not_count=[], corpus_docs=None, query_docs=None, IR_filename="", top_n_retrieval=20):
	'''
	:param kernel_matrix_test: shape: size = |test_samples| x |train_samples|
	:param train_labels:			  size = |train_samples| or |train_samples| x num_labels
	:param test_labels:			   size = |test_samples| or |test_samples| x num_labels
	:param list_percRetrieval:		list of fractions at which IR has to be calculated
	:param single_precision:		  True, if only one fraction is used
	:param label_type:				"single" or "multi"
	:param evaluation:				"strict" or "relaxed", only for 
	:return:
	'''
	#print('Computing IR prec......')

	if not len(test_labels) == len(kernel_matrix_test):
		print('mismatched samples in test_labels and kernel_matrix_test')
		exit()

	labels_to_count = []

	prec = []
	prec_label_wise = {label:[] for label in range(test_labels.shape[1])}

	if single_precision:
		vec_simIndexSorted = np.argsort(kernel_matrix_test, axis=1)[:, ::-1]
		prec_num_docs = np.floor(list_percRetrieval[0] * kernel_matrix_test.shape[1])
		#prec_num_docs = list_percRetrieval[0]
		vec_simIndexSorted_prec = vec_simIndexSorted[:, :int(prec_num_docs)]
		
		for counter, indices in enumerate(vec_simIndexSorted_prec):
			if label_type == "multi":
				classQuery = test_labels[counter, :]
				tr_labels = train_labels[indices, :]
			else:
				classQuery = test_labels[counter]
				tr_labels = train_labels[indices]
			list_percPrecision = np.zeros(len(list_percRetrieval))

			vec_goodLabel = compare_labels(tr_labels, classQuery, label_type=label_type, evaluation_type=evaluation, labels_to_count=labels_to_count)

			list_percPrecision[0] = np.sum(vec_goodLabel) / float(len(vec_goodLabel))

			prec += [list_percPrecision]

			#import pdb; pdb.set_trace()

			for i, label in enumerate(classQuery):
				if label == 1:
					prec_label_wise[i].append(list_percPrecision[0])
		
		if (corpus_docs is not None) and (query_docs is not None) and (IR_filename != ""):
			with open(IR_filename, "w") as f:
				for counter, indices in enumerate(vec_simIndexSorted[:, :top_n_retrieval]):
					f.write("Query\t::\t" + query_docs[counter] + "\n\n")
					#for index in indices[:int(kernel_matrix_test.shape[1] * 0.02)]:
					for index in indices[:top_n_retrieval]:
						f.write(str(kernel_matrix_test[counter, index]) + "\t::\t" + corpus_docs[index] + "\n")
					f.write("\n\n")
	else:
		vec_simIndexSorted = np.argsort(kernel_matrix_test, axis=1)[:, ::-1]
		for counter, indices in enumerate(vec_simIndexSorted):
			
			#indices = np.delete(indices, 0)
			
			if label_type == "multi":
				classQuery = test_labels[counter, :]
				tr_labels = train_labels[indices, :]
			else:
				classQuery = test_labels[counter]
				tr_labels = train_labels[indices]
			list_percPrecision = np.zeros(len(list_percRetrieval))

			vec_goodLabel = compare_labels(tr_labels, classQuery, label_type=label_type, evaluation_type=evaluation, labels_to_count=labels_to_count)

			list_totalRetrievalCount = []
			for frac in list_percRetrieval:
				list_totalRetrievalCount.append(np.floor(frac * kernel_matrix_test.shape[1]))
				#list_totalRetrievalCount.append(frac)

			countGoodLabel = 0
			for indexRetrieval, totalRetrievalCount in enumerate(list_totalRetrievalCount):
				if indexRetrieval == 0:
					countGoodLabel += np.sum(vec_goodLabel[:int(totalRetrievalCount)])
				else:
					countGoodLabel += np.sum(vec_goodLabel[int(lastTotalRetrievalCount):int(totalRetrievalCount)])

				list_percPrecision[indexRetrieval] = countGoodLabel / float(totalRetrievalCount)
				lastTotalRetrievalCount = totalRetrievalCount

			prec += [list_percPrecision]  # vec_simIndexSorted[:int(list_totalRetrievalCount[0])]
			
		if (corpus_docs is not None) and (query_docs is not None) and (IR_filename != ""):
			with open(IR_filename, "w") as f:
				for counter, indices in enumerate(vec_simIndexSorted[:, :top_n_retrieval]):
					f.write("Query\t::\t" + query_docs[counter] + "\n\n")
					#for index in indices[:int(kernel_matrix_test.shape[1] * 0.02)]:
					for index in indices[:top_n_retrieval]:
						f.write(str(kernel_matrix_test[counter, index]) + "\t::\t" + corpus_docs[index] + "\n")
					f.write("\n\n")
			

	prec = np.mean(prec, axis=0)

	return prec, prec_label_wise


def perform_classification_test_multi(train_data, test_data, c_list, classification_model="logistic", norm_before_classification=False):
	docVectors_train, train_labels = train_data
	docVectors_test, test_labels = test_data

	if norm_before_classification:
		mean = np.mean(np.vstack((docVectors_train, docVectors_test)), axis=0)
		std = np.std(np.vstack((docVectors_train, docVectors_test)), axis=0)

		docVectors_train = (docVectors_train - mean) / std
		docVectors_test = (docVectors_test - mean) / std

	## Classification Accuracy
	test_acc = []
	test_f1  = []
	
	for c in c_list:
		if classification_model == "logistic":
			clf = OneVsRestClassifier(LogisticRegression(C=c), n_jobs=5)
		elif classification_model == "svm":
			clf = OneVsRestClassifier(SVC(C=c, kernel='precomputed'))
		
		clf.fit(docVectors_train, train_labels)
		pred_test_labels = clf.predict(docVectors_test)

		acc_test = accuracy_score(test_labels, pred_test_labels)
		f1_test = precision_recall_fscore_support(test_labels, pred_test_labels, pos_label=None, average='macro')[2]

		test_acc.append(acc_test)
		test_f1.append(f1_test)

	return test_acc, test_f1


def perform_classification_test(train_data, test_data, c_list, classification_model="logistic", norm_before_classification=False):
	docVectors_train, train_labels = train_data
	docVectors_test, test_labels = test_data

	if norm_before_classification:
		mean = np.mean(np.vstack((docVectors_train, docVectors_test)), axis=0)
		std = np.std(np.vstack((docVectors_train, docVectors_test)), axis=0)

		docVectors_train = (docVectors_train - mean) / std
		docVectors_test = (docVectors_test - mean) / std

	## Classification Accuracy
	test_acc = []
	test_f1  = []
	
	for c in c_list:
		if classification_model == "logistic":
			clf = LogisticRegression(C=c)
		elif classification_model == "svm":
			clf = SVC(C=c, kernel='precomputed')
		
		clf.fit(docVectors_train, train_labels)
		pred_test_labels = clf.predict(docVectors_test)

		acc_test = accuracy_score(test_labels, pred_test_labels)
		f1_test = precision_recall_fscore_support(test_labels, pred_test_labels, pos_label=None, average='macro')[2]

		test_acc.append(acc_test)
		test_f1.append(f1_test)

	return test_acc, test_f1


def closest_docs_by_index(sim, order, query_vectors, n_docs):
	docs = []
	for i in range(len(query_vectors)):
		docs.append(order[:, i][0:n_docs])
	return np.array(docs)


def precision(label, predictions):
	if len(predictions):
		return float(
			len([x for x in predictions if label in x])
		) / len(predictions)
	else:
		return 0.0


def evaluate(
	corpus_vectors,
	query_vectors,
	corpus_labels,
	query_labels,
	recall=[0.02],
	num_classes=None,
	multi_label=False,
	query_docs=None,
	corpus_docs=None,
	IR_filename="",
	top_n_retrieval=20
):
	if multi_label:

		splitted_query_labels = [label[0].split(':') for label in query_labels]
		splitted_corpus_labels = [label[0].split(':') for label in corpus_labels]

		total_labels = []
		total_labels.extend(splitted_query_labels)
		total_labels.extend(splitted_corpus_labels)

		mlb = MultiLabelBinarizer(classes=[str(i) for i in range(num_classes)])
		mlb.fit([str(i) for i in range(num_classes)])

		query_one_hot_labels = mlb.transform(splitted_query_labels)
		corpus_one_hot_labels = mlb.transform(splitted_corpus_labels)

		similarity_matrix = pw.cosine_similarity(corpus_vectors, query_vectors).T
		if len(recall) == 1:
			single_precision = True
		else:
			single_precision = False

		results, results_label_wise = perform_IR_prec(similarity_matrix, corpus_one_hot_labels, query_one_hot_labels, list_percRetrieval=recall, single_precision=single_precision, label_type="multi", evaluation="relaxed", corpus_docs=corpus_docs, query_docs=query_docs, IR_filename=IR_filename, top_n_retrieval=top_n_retrieval)
	else:
		corpus_size = len(corpus_labels)
		query_size = len(query_labels)

		results = []
		results_label_wise = {}

		sim = pw.cosine_similarity(corpus_vectors, query_vectors)
		order = np.argsort(sim, axis=0)[::-1]

		for r in recall:
			n_docs = int((corpus_size * r) + 0.5)
			if not n_docs:
				results.append(0.0)
				continue
			
			closest = closest_docs_by_index(sim, order, query_vectors, n_docs)

			avg = 0.0
			for i in range(query_size):
				doc_labels = query_labels[i]
				doc_avg = 0.0
				for label in doc_labels:
					try:
						doc_avg += precision(label, corpus_labels[closest[i]])
					except:
						import pdb; pdb.set_trace()
				doc_avg /= len(doc_labels)
				avg += doc_avg
			avg /= query_size
			results.append(avg)
	
	return results
