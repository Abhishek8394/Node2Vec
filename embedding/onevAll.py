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

	def next_batch(self):
		batch = []
		for i in self.recList:
			b = self.create_record(i)
			batch.append(b)
		return batch


def record2string(rec):
	nds = [str(x) for x in rec['nodes']]
	lbls = [str(x[0]) for x in rec['labels']]
	return " ".join([str(x) for x in zip(nds,lbls)])

def batch2string(batch):
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


def executeTraining(train_dataset_merged, valid_dataset_merged, num_epochs, batch_size, vocabulary_size, embedding_size, num_labels, hidden_size, 
					summary_frequency, embeddings_file, log_directory, num_checkpoints = 5):
	graph = tf.Graph()
	with graph.as_default():
		tg = TrainingGraph(vocabulary_size, embedding_size, num_labels, hidden_size)
		precision_tf = tf.placeholder(shape=[], dtype=tf.float32,name='precision')
		recall_tf = tf.placeholder(shape=[], dtype=tf.float32,name='recall')
		f1_tf = tf.placeholder(shape=[], dtype=tf.float32,name='f1')

	train_batch = BatchGenerator(train_dataset_merged, batch_size, num_labels)	
	
	precision_summary = tf.summary.scalar('precision_summary',precision_tf)
	recall_summary = tf.summary.scalar('recall_summary',recall_tf)
	f1_summary = tf.summary.scalar('f1_summary',f1_tf)
	stat_summary = tf.summary.merge([precision_summary, recall_summary, f1_summary])
	stat_dict={}

	summary_directory = os.path.join(log_directory,"summaries")
	train_log_directory = os.path.join(summary_directory,"train")
	valid_log_directory = os.path.join(summary_directory, "valid")
	train_model_directory = os.path.join(log_directory,"models")
	train_model_file = os.path.join(train_model_directory, 'checkpoint')
	utility.makeDir(train_log_directory)
	utility.makeDir(train_model_directory)
	utility.makeDir(valid_log_directory)
	train_summary_writer = tf.summary.FileWriter(train_log_directory, graph = graph)

	maxLenRecord = max(train_dataset_merged,key = lambda x:len(train_dataset_merged[x]))
	maxLen = len(train_dataset_merged[maxLenRecord]) 
	validMaxLenRecord = max(valid_dataset_merged,key = lambda x:len(valid_dataset_merged[x]))
	validMaxLen = len(train_dataset_merged[validMaxLenRecord]) 

	num_iters = (maxLen // batch_size) * num_epochs	
	feed_dict={}
	embeddings = utility.loadEmbeddings(embeddings_file)
	feed_dict[tg.embeddings] = embeddings
	print("Will take {} iters".format(num_iters))
	# average loss of the system as whole
	overall_avg_loss = 0.0
	with tf.Session(graph=graph) as session:
		session.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables(),max_to_keep = num_checkpoints)
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
				feed_dict[tg.classifiers[j].inp_x] = batch[j]['nodes']
				feed_dict[tg.classifiers[j].labels] = batch[j]['labels']
				cl, prediction, _, loss_summary, l2_summary = session.run([tg.classifiers[j].loss, tg.classifiers[j].prediction, tg.classifiers[j].optimizer,
												   tg.classifiers[j].loss_summary, tg.classifiers[j].l2_summary], feed_dict=feed_dict)
				net_loss+=cl
				summaries_calculated.append(loss_summary)
				summaries_calculated.append(l2_summary)
				classifier_ops.append(prediction)

			pred_hot_vec = []
			for j in range(len(classifier_ops)):
				pred_hot_vec.append(hotEncodeDistribution(classifier_ops[j]))

			del classifier_ops			
			for j in range(len(summaries_calculated)):
				train_summary_writer.add_summary(summaries_calculated[j], i)

			f1, precision, recall = get_accuracy(pred_hot_vec, batch)
			stat_dict[precision_tf] = precision
			stat_dict[recall_tf] = recall
			stat_dict[f1_tf] = f1
			pre, rec, ef1, ss = session.run([precision_tf, recall_tf, f1_tf, stat_summary],feed_dict = stat_dict)
			train_summary_writer.add_summary(ss,i)

			print("step {}/{}: loss: {}, f1:{}".format(i,num_iters,net_loss, f1))
			if i%summary_frequency==0:
				save_loc = saver.save(session, train_model_file, global_step = i)
				print("Saving model at {}".format(save_loc))
				for j in range(len(pred_hot_vec)):
					print(pred_hot_vec[j],"--",batch[j]['labels'])
					print()
		
		valid_batch = BatchGenerator(valid_dataset_merged, validMaxLen, num_labels)
		

parser = argparse.ArgumentParser()
parser.add_argument("--embedding-file",help="embeddings txt file to read from", required=True)
args = parser.parse_args()

res = loadDataset("../data/nodes.csv","../data/groups.csv","../data/balanced-group-edges.csv")
dataset = res['node2labels']
nodes=res['nodes']
labels = res['labels']
split_ratio = 0.75
num_epochs = 1			
batch_size = 5
hidden_size = 15
embedding_size = 128
summary_frequency = 10
num_labels = len(labels)

train_dataset = dataset
train_dataset_merged, valid_dataset_merged = collectNodesAndLabels(train_dataset,True, split_ratio)

timestamp = int(time.time())
log_directory = os.path.join("classifier_runs",str(timestamp))
utility.makeDir(log_directory)
write_metadata = os.path.join(log_directory,"metadata.txt")
writeMeta(write_metadata, args.embedding_file, hidden_size)
executeTraining(train_dataset_merged, valid_dataset_merged, num_epochs, batch_size, len(nodes), embedding_size, len(labels), hidden_size, 
					summary_frequency, args.embedding_file, log_directory)
