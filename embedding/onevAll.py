import tensorflow as tf
import numpy as np
import random
import os
from dataHandler import node2id, id2node, label2id, id2label
import utility
import argparse
import time

"""
nodefile -> just a list of all possible nodes.
labelfile -> just a list of all possible labels.
node2labelFile -> label assignments for node. 
"""
def loadDataset(nodeFile, labelFile, node2labelFile):
	nodes={}			# node to id mappings
	labels={}			# label to id mappings
	node2labels=[]		# list of node-label tuples
	with open(nodeFile,"r") as f:
		for j in f:
			nd = int(j)
			nodes[nd] = len(nodes)
	with open(labelFile, "r") as f:
		for j in f:
			lbl = int(j)
			labels[lbl] = len(labels)
	with open(node2labelFile, "r") as f:
		for j in f:
			entry = [int(x) for x in j.split(",")]
			node2labels.append((entry[1], entry[0], entry[2]))
	reverseNodes = dict(zip(nodes.values(), nodes.keys()))
	reverseLabels = dict(zip(labels.values(), labels.keys()))
	return {"nodes":nodes, "labels":labels, "node2labels":node2labels, "reverseNodes":reverseNodes, "reverseLabels":reverseLabels}

def writeMeta(meta_file, embedding_file, hidden_size):
	j = {'embedding_file':embedding_file, 'hidden_size':hidden_size}
	with open(meta_file,"w") as f:
		f.write(str(j))

"""
returns 2 dictionaries, keys => labels, values=> tuples of (nodes, 1 if label else 0)
1st dictionary = training set
2nd dictionary = validation set
The split is done on a per label basis, i.e the number of entries for each label are split according to the split ratio.
"""
def collectNodesAndLabels(node2labels, should_shuffle, split_ratio=1):
	nl1 = {}
	nl2 = {}
	for n,l,f in node2labels:
		if l not in nl1:
			nl1[l]=[]
			nl2[l] =[]
		if f==1:
			nl1[l].append((n,1))
		else:
			nl1[l].append((n,0))
	for i in nl1:
		if should_shuffle:
			random.shuffle(nl1[i])
		if split_ratio!=1:
			cutoff = int(len(nl1[i]) * split_ratio)
			train = nl1[i][:cutoff]
			valid = nl1[i][cutoff:]
			nl1[i] = train
			nl2[i] = valid
	return nl1, nl2


def hotEncodeDistribution(probabilities):
	p=[]
	for i in probabilities:
		p.append(1.0 if i[0]>=i[1] else 0.0)
	return p 

def hotEncode(num_labels, labels):
	hotVec = []
	for i in range(num_labels):
		hotVec.append([0.0,1.0])
	for i in labels:
		hotVec[label2id(i)] = [1.0,0.0]
	return hotVec

def hotDecode(hotVec):
	s = [id2label(i) for i in range(len(hotVec)) if hotVec[i][0]==1]
	return s

"""
Each row of a batch will be : [batch] * len(labels)
batch = {nodes: [list of node ids], labels:[entry for corresponding nodes]}; [1,0] if yes, [0,1] if no
TODO: workaround for large number of labels
"""
class BatchGenerator(object):
	def __init__(self, node2labels, batch_size, num_labels):
		self.batch_size = batch_size
		self.dataset = node2labels
		self.recList = sorted([label2id(x) for x in node2labels.keys()])	# list of records, in this case all the labels.
		# each element refers to a cursor for a record. 
		self.cursors = [0] * len(self.recList)
		self.num_labels = num_labels

	# pick batch_size number of training tuples from the dataset for classifer with id 'index'
	def create_record(self,index):
		nodes=[0] * self.batch_size
		labels=[0] * self.batch_size
		lbl = id2label(index)	# for indexing in dataset
		for i in range(self.batch_size):
			cur = self.cursors[index]
			entry = self.dataset[lbl][cur]
			nodes[i] = node2id(entry[0])
			if entry[1]==1:
				labels[i] = [1,0]
			else:
				labels[i] = [0,1]
			self.cursors[index] = (self.cursors[index] + 1) % len(self.dataset[lbl])
		return {'nodes':nodes, 'labels':labels}

	# create a batch for each classifier and return them all as a list.
	def next_batch(self):
		batch = []
		for i in self.recList:
			b = self.create_record(i)
			batch.append(b)
		return batch

	# return batch for just one classifier.
	def classifier_next_batch(self, classifier_id):
		return self.create_record(classifier_id)

	def resetCursors(self, classifier_id):
		self.cursors[classifier_id] = 0

	def resetAllCursors(self):
		for i in range(len(self.cursors)):
			self.cursors[i] = 0


def record2string(rec):
	nds = [str(id2node(x)) for x in rec['nodes']]
	lbls = [str(x[0]) for x in rec['labels']]
	return " ".join([str(x) for x in zip(nds,lbls)])

def batch2string(batch):
	if batch is not list:
		return record2string(batch)
	s = []
	for b in batch:
		s.append(record2string(b))
	return "\n".join(s)


class classifier_core_layer(object):
	def __init__(self, _id,embedding_size, hidden_size, embeddings, l2_lambda=0):
		self.id = _id
		self.inp_x = tf.placeholder(shape=[None], dtype=tf.int32, name='inp_x')
		self.embed = tf.nn.embedding_lookup(embeddings, self.inp_x)
		self.labels = tf.placeholder(shape=[None,2], dtype=tf.float32, name='labels')
		# For entirety of current batch
		self.f1 = tf.placeholder(shape=[], dtype=tf.float32, name='f1')
		self.precision = tf.placeholder(shape=[], dtype=tf.float32, name='precision')
		self.recall = tf.placeholder(shape=[], dtype=tf.float32, name='recall')
		# Placeholders for calculating average across several batches
		self.avg_f1 = tf.placeholder(shape=[], dtype=tf.float32, name='avg_f1')
		self.avg_precision = tf.placeholder(shape=[], dtype=tf.float32, name='avg_precision')
		self.avg_recall = tf.placeholder(shape=[], dtype=tf.float32, name='avg_recall')
		self.global_step = tf.Variable(0)

		self.w1 = tf.Variable(tf.truncated_normal([embedding_size, hidden_size],-0.1,0.1), dtype=tf.float32, name="weight1")
		self.b1 = tf.Variable(tf.zeros([hidden_size]), dtype=tf.float32, name="bias1")
		self.o1 = tf.sigmoid(tf.matmul(self.embed,self.w1) + self.b1)
		# first output is yes, second is no.
		self.w2 = tf.Variable(tf.truncated_normal([hidden_size,2],-0.1,0.1), dtype=tf.float32, name="weight2")
		self.b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name="bias2")
		self.logits = tf.matmul(self.o1, self.w2) + self.b2

		self.prediction = tf.nn.softmax(self.logits, name="prediction")
		self.l2_loss = tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)) + (l2_lambda * self.l2_loss)
		# loss = tf.reduce_mean(-(labels * tf.log(self.prediction + 1e-7) + (1-labels)*tf.log(1-self.prediction + 1e-7))) + (l2_lambda * self.l2_loss) 
		# self.loss = tf.identity(loss, name="loss")
		self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
		self.l2_summary = tf.summary.scalar("l2_summary", self.l2_loss)
		self.loss_summary = tf.summary.scalar("loss_summary", self.loss) 	
		self.f1_summary = tf.summary.scalar("f1_summary", self.f1) 
		self.precision_summary = tf.summary.scalar("precision_summary", self.precision) 
		self.recall_summary = tf.summary.scalar("recall_summary", self.recall) 
		self.avg_f1_summary = tf.summary.scalar("avg_f1_summary", self.avg_f1) 
		self.avg_precision_summary = tf.summary.scalar("avg_precision_summary", self.avg_precision) 
		self.avg_recall_summary = tf.summary.scalar("avg_recall_summary", self.avg_recall) 		


class TrainingGraph(object):
	def __init__(self, vocabulary_size, embedding_size, num_labels, hidden_size):
		# self.inp_x = tf.placeholder(shape=[None], dtype=tf.int32, name="inp_x")
		# self.labels = tf.placeholder(shape=[num_labels,None,2], dtype=tf.float32, name="labels")
		self.embeddings = tf.placeholder(shape=[vocabulary_size,embedding_size], dtype=tf.float32,name="embeddings")
		self.classifiers=[]
		classify_summaries=[]
		for i in range(num_labels):
			with tf.variable_scope('label_classifier-'+str(i)):
				csfr = classifier_core_layer(i, embedding_size, hidden_size, self.embeddings, 0.01)					
				self.classifiers.append(csfr)
				classify_summaries.append(csfr.loss_summary)
				classify_summaries.append(csfr.l2_summary)
		self.all_summaries = tf.summary.merge(classify_summaries)

# crude count of amount of labels predicted accurately. 
# Not accuracy exactly, but for a rough idea.
def get_accuracy(pred, labels):
	acc=0
	n = 0
	c=0
	net_pos = 0
	net_neg = 0
	TP = 0 #true positive
	FN = 0 # false negative
	FP = 0 # false positive
	TN = 0 # true negative
	# not looping on labels because this makes testing easy.
	for i in range(len(pred)):
		# c = 0
		for j in range(len(pred[0])):
			if labels[i]['labels'][j][0]==1:
				if pred[i][j]==1:
					TP+=1
				else:
					FN+=1
				net_pos+=1
			else:
				if pred[i][j]==0:
					TN+=1
				else:
					FP+=1
				net_neg+=1
	precision = float(TP) / (TP + FP + 1e-7)
	recall = float(TP) / (TP + FN + 1e-7)
	f1 = 2 * precision * recall / (precision + recall + 1e-7)
	# print("TP: {}, TN:{}, FP:{}, FN:{}".format(TP,TN,FP,FN))
	return f1,precision,recall

def createInpOutListsFromBatch(batch):
	inp_x = []
	inp_y=[]
	# separate batch into inp and op
	for k in batch:
		inp_x.append(k[0])
		inp_y.append(k[1])
	return inp_x, inp_y

def generateLogDirectories(log_directory):
	summary_directory = os.path.join(log_directory,"summaries")
	train_log_directory = os.path.join(summary_directory,"train")
	valid_log_directory = os.path.join(summary_directory, "valid")
	train_model_directory = os.path.join(log_directory,"models")
	train_model_file = os.path.join(train_model_directory, 'checkpoint')
	utility.makeDir(train_log_directory)
	utility.makeDir(train_model_directory)
	utility.makeDir(valid_log_directory)
	return {
			'summary_directory':summary_directory,
			'train_log_directory':train_log_directory,
			'valid_log_directory':valid_log_directory,
			'train_model_directory':train_model_directory,
			'train_model_file':train_model_file
			}

def executeTrainStep(session, classifierId, trainingGraph, inp_x, labels,  feed_dict, is_training):
	tg = trainingGraph
	# classifier_ops = []
	summaries_calculated = []
	net_loss = 0.0
	cl = None 	# classifier loss
	prediction = None 
	loss_summary = None
	l2_summary = None
	j = classifierId
	feed_dict[tg.classifiers[j].inp_x] = inp_x
	feed_dict[tg.classifiers[j].labels] = labels
	if is_training:
		cl, prediction, _, loss_summary, l2_summary = session.run([tg.classifiers[j].loss, tg.classifiers[j].prediction, tg.classifiers[j].optimizer,
										   tg.classifiers[j].loss_summary, tg.classifiers[j].l2_summary], feed_dict=feed_dict)
	else:
		cl, prediction, loss_summary, l2_summary = session.run([tg.classifiers[j].loss, tg.classifiers[j].prediction,
										   tg.classifiers[j].loss_summary, tg.classifiers[j].l2_summary], feed_dict=feed_dict)
	net_loss+=cl
	summaries_calculated.append(loss_summary)
	summaries_calculated.append(l2_summary)
	# classifier_ops.append(prediction)
	return {'classifier_ops':prediction, 'summaries_calculated':summaries_calculated, 'net_loss': net_loss}

def executeTraining(train_dataset_merged, valid_dataset_merged, num_epochs, batch_size, vocabulary_size, embedding_size, num_labels, hidden_size, summary_frequency, embeddings_file, log_directories, num_checkpoints = 5):
	graph = tf.Graph()
	with graph.as_default():
		tg = TrainingGraph(vocabulary_size, embedding_size, num_labels, hidden_size)
		precision_tf = tf.placeholder(shape=[], dtype=tf.float32,name='precision')
		recall_tf = tf.placeholder(shape=[], dtype=tf.float32,name='recall')
		f1_tf = tf.placeholder(shape=[], dtype=tf.float32,name='f1')
		macro_f1_tf = tf.placeholder(shape=[], dtype=tf.float32,name='macro_f1')

	train_batch = BatchGenerator(train_dataset_merged, batch_size, num_labels)	
	valid_batch = BatchGenerator(valid_dataset_merged, batch_size, num_labels)

	precision_summary = tf.summary.scalar('precision_summary',precision_tf)
	recall_summary = tf.summary.scalar('recall_summary',recall_tf)
	f1_summary = tf.summary.scalar('f1_summary',f1_tf)
	macro_f1_summary = tf.summary.scalar('macro_f1_summary', macro_f1_tf)
	stat_summary = tf.summary.merge([precision_summary, recall_summary, f1_summary])
	stat_dict={}

	summary_directory = log_directories['summary_directory']
	train_log_directory = log_directories['train_log_directory']
	valid_log_directory = log_directories['valid_log_directory']
	train_model_directory = log_directories['train_model_directory']
	train_model_file = log_directories['train_model_file']
	
	train_summary_writer = tf.summary.FileWriter(train_log_directory, graph = graph)
	valid_summary_writer = tf.summary.FileWriter(valid_log_directory, graph = graph)

	maxLenRecord = max(train_dataset_merged,key = lambda x:len(train_dataset_merged[x]))
	maxLen = len(train_dataset_merged[maxLenRecord]) 
	validMaxLenRecord = max(valid_dataset_merged,key = lambda x:len(valid_dataset_merged[x]))
	validMaxLen = len(train_dataset_merged[validMaxLenRecord]) 

	num_iters = (maxLen // batch_size) * num_epochs	
	feed_dict={}
	embeddings = utility.loadEmbeddings(embeddings_file)
	feed_dict[tg.embeddings] = embeddings
	precision_accum = 0.0
	recall_accum = 0.0
	print("Will take {} iters".format(num_iters))
	# average loss of the system as whole
	overall_avg_loss = 0.0
	with tf.Session(graph=graph) as session:
		session.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables(),max_to_keep = num_checkpoints)
		valid_test = ValidationTest(graph, valid_batch, valid_summary_writer)
		for i in range(num_iters):
			batch = train_batch.next_batch()
			num_classifiers_to_test = len(tg.classifiers)
			# store loss and predictions for each classifier. Loss is for general overall loss average
			# calculation of system as a whole, storing labels to calculate accuracy.
			classifier_ops = []
			net_loss = 0.0
			f1_score = 0.0
			summaries_calculated = []
			for j in range(num_classifiers_to_test):
				op = executeTrainStep(session, j, tg, batch[j]['nodes'], batch[j]['labels'], feed_dict, True)
				classifier_ops.append(op['classifier_ops'])
				net_loss += op['net_loss']
				summaries_calculated.extend(op['summaries_calculated'])
			pred_hot_vec = []
			for j in range(len(classifier_ops)):
				pred_hot_vec.append(hotEncodeDistribution(classifier_ops[j]))

			del classifier_ops			
			for j in range(len(summaries_calculated)):
				train_summary_writer.add_summary(summaries_calculated[j], i)

			f1, precision, recall = get_accuracy(pred_hot_vec, batch)
			precision_accum += precision
			recall_accum += recall
			stat_dict[precision_tf] = precision
			stat_dict[recall_tf] = recall
			stat_dict[f1_tf] = f1
			pre, rec, ef1, ss = session.run([precision_tf, recall_tf, f1_tf, stat_summary],feed_dict = stat_dict)
			train_summary_writer.add_summary(ss,i)
			train_summary_writer.flush()
			print("step {}/{}: loss: {}, f1:{}".format(i,num_iters,net_loss, f1))
			if i%summary_frequency==0:
				save_loc = saver.save(session, train_model_file, global_step = i)
				valid_test.run_validation(session, tg, num_classifiers_to_test, feed_dict)
				valid_summary_writer.flush()
				print("Saving model at {}".format(save_loc))
				for j in range(len(pred_hot_vec)):
					print(pred_hot_vec[j],"--",batch[j]['labels'])
					print()
		valid_test.run_validation(session, tg, num_classifiers_to_test, feed_dict)
		precision_accum /= num_iters
		recall_accum /= num_iters
		macro_f1 = 2 * precision_accum * recall_accum / (precision_accum + recall_accum + 1e-7)
		print("Training Macro F1 score: {}".format(macro_f1))
		macro_f1_summ = session.run([macro_f1_summary],feed_dict={macro_f1_tf: macro_f1})
		train_summary_writer.add_summary(macro_f1_summ, 0)
		valid_macro_prec = valid_test.precision_accum / valid_test.global_counter
		valid_macro_rec = valid_test.recall_accum / valid_test.global_counter
		valid_macro_f1 = 2 * valid_macro_prec * valid_macro_rec / (valid_macro_prec + valid_macro_rec + 1e-7)
		print("Validation Macro F1 score: {}".format(valid_macro_f1))
		macro_f1_summ = session.run([macro_f1_summary],feed_dict={macro_f1_tf: valid_macro_f1})
		valid_summary_writer.add_summary(macro_f1_summ, 0)
		valid_summary_writer.flush()
		train_summary_writer.flush()

class ValidationTest(object):

	def __init__(self, graph, valid_batch, valid_summary_writer):
		self.global_counter = 0
		with graph.as_default():
			self.avg_f1 = tf.placeholder(shape=[], dtype=tf.float32, name='avg_f1')
			self.avg_prec = tf.placeholder(shape=[], dtype=tf.float32, name='avg_prec')
			self.avg_rec = tf.placeholder(shape=[], dtype=tf.float32, name='avg_rec')
			self.avg_f1_summary = tf.summary.scalar('avg_f1', self.avg_f1)
			self.avg_prec_summary = tf.summary.scalar('avg_prec', self.avg_prec)
			self.avg_rec_summary = tf.summary.scalar('avg_rec', self.avg_rec)
		self.precision_accum = 0.0
		self.recall_accum = 0.0
		self.valid_batch = valid_batch
		self.valid_summary_writer =  valid_summary_writer
		self.avg_f1,self.avg_prec,self.avg_rec = 0.0, 0.0, 0.0

	def resetCounter(self):
		self.global_counter = 0

	def run_validation(self, session, tg, num_classifiers_to_test, feed_dict, specific_classifiers=[]):
		
		run_on_classifiers = range(num_classifiers_to_test) if len(specific_classifiers)==0 else specific_classifiers 
		for j in run_on_classifiers:
			print("Running validation tests on classifier " + str(j))
			clsfr = tg.classifiers[j]
			self.valid_batch.batch_size = len(self.valid_batch.dataset[id2label(j)])
			num_iters = 1 #len(self.valid_batch.dataset[id2label(j)]) // self.valid_batch.batch_size 
			self.valid_batch.resetCursors(j)
			# if num_iters==0:
			# 	print("classifier {} has no validation data!".format(j))
			# 	continue
			# average f1, precision, recall across all batches
			f1_s,prec_s,rec_s = 0.0, 0.0, 0.0
			for i in range(num_iters):
				batch = self.valid_batch.classifier_next_batch(j)
				op = executeTrainStep(session, j, tg, batch['nodes'], batch['labels'], feed_dict, True)
				pred=hotEncodeDistribution(op['classifier_ops'])

				f1, prec, rec = get_accuracy([pred], [batch])
				f1_summ, prec_summ, rec_summ = session.run([clsfr.f1_summary,clsfr.precision_summary, clsfr.recall_summary],
															feed_dict={clsfr.precision:prec, clsfr.f1:f1, clsfr.recall:rec})
				# map(lambda x:self.valid_summary_writer.add_summary(x,i),[f1_summ, prec_summ, rec_summ])
				self.valid_summary_writer.add_summary(f1_summ,i)
				self.valid_summary_writer.add_summary(prec_summ,i)
				self.valid_summary_writer.add_summary(rec_summ,i)
				f1_s += f1
				prec_s += prec
				rec_s += rec
			self.precision_accum+=prec_s
			self.recall_accum+=rec
			f1_s /= num_iters
			prec_s /= num_iters
			rec_s /= num_iters
			f1_summ, prec_summ, rec_summ = session.run([clsfr.avg_f1_summary, clsfr.avg_precision_summary, clsfr.avg_recall_summary],
														feed_dict={clsfr.avg_f1:f1_s, clsfr.avg_precision:prec_s, clsfr.avg_recall:rec_s})
			print("f1: {}, precision: {}, recall: {}".format(f1_s, prec_s, rec_s))
			# map(lambda x:self.valid_summary_writer.add_summary(x,self.global_counter),[f1_summ, prec_summ, rec_summ])
			self.valid_summary_writer.add_summary(f1_summ,self.global_counter)
			self.valid_summary_writer.add_summary(prec_summ,self.global_counter)
			self.valid_summary_writer.add_summary(rec_summ,self.global_counter)
			self.avg_f1+=f1_s
			self.avg_prec+=prec_s
			self.avg_rec+=rec_s
		if len(specific_classifiers)==0:
			avg_f1 = self.avg_f1 / num_classifiers_to_test
			avg_prec = self.avg_prec / num_classifiers_to_test
			avg_rec = self.avg_rec / num_classifiers_to_test
			f1_summ, prec_summ, rec_summ = session.run([self.avg_f1_summary, self.avg_prec_summary, self.avg_rec_summary],
														feed_dict={self.avg_f1:avg_f1, self.avg_prec:avg_prec, self.avg_rec:avg_rec})
			self.valid_summary_writer.add_summary(f1_summ,self.global_counter)
			self.valid_summary_writer.add_summary(prec_summ,self.global_counter)
			self.valid_summary_writer.add_summary(rec_summ,self.global_counter)
			print("avg f1: {}, avg prec: {}, avg rec: {}".format(avg_f1, avg_prec, avg_rec))
		self.global_counter+=1


def trainSingleClassifier(classifierId, graph, session, trainingGraph, dataset, batch_size, embeddings, batchGen, num_epochs, 
						  train_summary_writer, saver, train_model_file, is_training, valid_test, summary_frequency=-1):
	tg = trainingGraph
	with graph.as_default():
		precision_tf = tf.placeholder(shape=[], dtype=tf.float32,name='precision')
		recall_tf = tf.placeholder(shape=[], dtype=tf.float32,name='recall')
		f1_tf = tf.placeholder(shape=[], dtype=tf.float32,name='f1')
		macro_f1_tf = tf.placeholder(shape=[], dtype=tf.float32,name='macro_f1')

	precision_summary = tf.summary.scalar('precision_summary',precision_tf)
	recall_summary = tf.summary.scalar('recall_summary',recall_tf)
	f1_summary = tf.summary.scalar('f1_summary',f1_tf)
	macro_f1_summary = tf.summary.scalar('macro_f1_summary', macro_f1_tf)
	stat_summary = tf.summary.merge([precision_summary, recall_summary, f1_summary, macro_f1_summary])
	stat_dict={}

	recordLength = len(dataset[id2label(classifierId)])
	num_iters = (recordLength // batch_size) * num_epochs	
	feed_dict={}
	feed_dict[tg.embeddings] = embeddings
	classifier_ops = []
	summaries=[]
	print("Classifier {} will take {} iters".format(classifierId, num_iters))
	for i in range(num_iters):
		batch = batchGen.classifier_next_batch(classifierId)
		op = executeTrainStep(session, classifierId, tg, batch['nodes'], batch['labels'], feed_dict, is_training)
		net_loss = op['net_loss']
		pred = [hotEncodeDistribution(op['classifier_ops'])]

		f1 = get_accuracy(pred, [batch])
		print("step: {} loss:{} f1:{}".format(i,net_loss,f1))
		summaries = op['summaries_calculated']
		for j in range(len(summaries)):
			train_summary_writer.add_summary(summaries[j], i)
		if summary_frequency > 0 and i%summary_frequency == 0:
			save_loc = saver.save(session, train_model_file, global_step=i)
			valid_test.run_validation(session, tg, 1, feed_dict, [classifierId])
			print(pred,batch['labels'])
	valid_test.run_validation(session, tg, 1, feed_dict, [classifierId])


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--embedding-file",help="embeddings txt file to read from", required=True)
	parser.add_argument("--meta-file",help="config file used when training embeddings", required=True)
	parser.add_argument("--config-file",help="config file for training the classifiers", required=True)
	parser.add_argument("--separate-train", help="train each classifier one by one", action='store_true')
	args = parser.parse_args()

	embedMeta = utility.ConfigProvider(args.meta_file)
	config = utility.ConfigProvider(args.config_file)
	nodeFile = config.getOption('classifier_nodeFile')		# list of nodes.
	labelFile = config.getOption('classifier_labelFile')	# list of labels
	dataFile = config.getOption('classifier_trainingFile')	# list of node to label data for learning.
	res = loadDataset(nodeFile, labelFile , dataFile)
	dataset = res['node2labels']
	nodes=res['nodes']
	labels = res['labels']
	split_ratio = config.getOption('classifier_split_ratio')
	num_epochs = config.getOption('classifier_num_epochs')
	batch_size = config.getOption('classifier_batch_size')
	hidden_size = config.getOption('classifier_hidden_size')
	summary_frequency = config.getOption('classifier_summary_frequency')
	embedding_size = embedMeta.getOption('embedding_size')	# reads from metadata file.
	num_labels = len(labels)
	vocabulary_size = len(nodes)

	# split dataset among training and validation sets.
	train_dataset = dataset
	train_dataset_merged, valid_dataset_merged = collectNodesAndLabels(train_dataset,True, split_ratio)

	# create directories for logging.
	timestamp = int(time.time())
	log_directory = os.path.join("classifier_runs",str(timestamp))
	utility.makeDir(log_directory)
	log_directories = generateLogDirectories(log_directory)
	# write metadata
	write_metadata = os.path.join(log_directory,"metadata.txt")
	writeMeta(write_metadata, args.embedding_file, hidden_size)
	utility.copyFile(args.config_file,os.path.join(log_directory, 'classify_config.txt'))
	if not args.separate_train:
		executeTraining(train_dataset_merged, valid_dataset_merged, num_epochs, batch_size, len(nodes), embedding_size, len(labels), hidden_size, 
							summary_frequency, args.embedding_file, log_directories)	
	else:
		# train each classifier individually.
		graph = tf.Graph()
		with graph.as_default():
			trainingGraph = TrainingGraph(vocabulary_size, embedding_size, num_labels, hidden_size)
			embeddings = utility.loadEmbeddings(args.embedding_file) 
			batchGen = BatchGenerator(train_dataset_merged, batch_size, num_labels)
			valid_batch = BatchGenerator(valid_dataset_merged, batch_size, num_labels)
		train_model_file = log_directories['train_model_file']
		with tf.Session(graph=graph) as session:
			session.run(tf.global_variables_initializer())
			saver = tf.train.Saver(tf.global_variables(),max_to_keep = 5)
			valid_summary_writer = tf.summary.FileWriter(log_directories['valid_log_directory'], graph = graph)
			train_summary_writer = tf.summary.FileWriter(log_directories['train_log_directory'], graph=graph)
			valid_test = ValidationTest(graph, valid_batch, valid_summary_writer)
			for i in range(num_labels):
				trainSingleClassifier(i, graph, session, trainingGraph, train_dataset_merged, batch_size, embeddings, batchGen, num_epochs, 
									  train_summary_writer, saver, train_model_file, True, valid_test, 50)
