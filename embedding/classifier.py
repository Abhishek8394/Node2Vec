import tensorflow as tf
import numpy as np
import random
import os
from dataHandler import node2id, id2node, label2id, id2label
import utility
import argparse

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
			node2labels.append((entry[0], entry[1]))
	reverseNodes = dict(zip(nodes.values(), nodes.keys()))
	reverseLabels = dict(zip(labels.values(), labels.keys()))
	return {"nodes":nodes, "labels":labels, "node2labels":node2labels, "reverseNodes":reverseNodes, "reverseLabels":reverseLabels}

def collectNodesAndLabels(node2labels):
	nl = {}
	for n,l in node2labels:
		if n not in nl:
			nl[n]=[]
		if n==4281:
			print(l)
		nl[n].append(l)
	return nl

def hotEncodeDistribution(probabilities):
	p=[]
	for i in probabilities:
		p.append(1.0 if i[0]>=0.5 else 0.0)
	return p 

def hotEncode(num_labels, labels):
	hotVec = [0.0] * num_labels
	for i in labels:
		hotVec[label2id(i)] = 1.0
	return hotVec

def hotDecode(hotVec):
	s = [id2label(i) for i in range(len(hotVec)) if hotVec[i]>=0.5]
	return s

"""
Each row of a batch will be : [node_id,[hot encoding of labels]]
TODO: workaround for large number of labels
"""
class BatchGenerator(object):
	def __init__(self, node2labels, batch_size, num_labels):
		self.batch_size = batch_size
		self.dataset = node2labels
		self.recList = list(node2labels.keys())
		partition_size = len(self.recList) // batch_size
		self.cursors = [i*partition_size for i in range(batch_size)]
		self.num_labels = num_labels

	def create_record(self,index):
		node = self.recList[self.cursors[index]]
		rec_labels = self.dataset[self.recList[self.cursors[index]]]
		label = hotEncode(self.num_labels, rec_labels)
		self.cursors[index] = (self.cursors[index] + 1) % len(self.dataset)
		node = node2id(node)
		return [node, label]

	def next_batch(self):
		batch = []
		for i in range(self.batch_size):
			b = self.create_record(i)
			batch.append(b)
		return batch

def record2string(rec):
	decodeLabels = hotDecode(rec[1])
	lbls = [str(x) for x in decodeLabels]
	return  str(id2node(rec[0]))+": "+ ",".join(lbls)

def batch2string(batch):
	s = []
	for b in batch:
		s.append(record2string(b))
	return "\n".join(s)


class classifier_core_layer(object):
	def __init__(self, embedding_size, hidden_size, embed, labels, l2_lambda=0):
		self.w1 = tf.Variable(tf.truncated_normal([embedding_size, hidden_size],-0.1,0.1), dtype=tf.float32, name="weight1")
		self.b1 = tf.Variable(tf.zeros([hidden_size]), dtype=tf.float32, name="bias1")
		self.o1 = tf.sigmoid(tf.matmul(embed,self.w1) + self.b1)
		self.w2 = tf.Variable(tf.truncated_normal([hidden_size,1],-0.1,0.1), dtype=tf.float32, name="weight2")
		self.b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name="bias2")
		self.logits = tf.matmul(self.o1, self.w2) + self.b2
		# print(self.o1.shape,self.logits.shape)
		# return
		labels = tf.reshape(labels,shape=[-1,1])
		self.prediction = tf.sigmoid(self.logits, name="prediction")
		self.l2_loss = tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2)
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=labels)) + (l2_lambda * self.l2_loss)
		# loss = tf.reduce_mean(-(labels * tf.log(self.prediction + 1e-7) + (1-labels)*tf.log(1-self.prediction + 1e-7))) + (l2_lambda * self.l2_loss) 
		# self.loss = tf.identity(loss, name="loss")
		self.optimizer = tf.train.GradientDescentOptimizer(1e-2).minimize(self.loss)
		self.l2_summary = tf.summary.scalar("l2_summary", self.l2_loss)
		self.loss_summary = tf.summary.scalar("loss_summary", self.loss)

class TrainingGraph(object):
	def __init__(self, vocabulary_size, embedding_size, num_labels, hidden_size):
		self.inp_x = tf.placeholder(shape=[None], dtype=tf.int32, name="inp_x")
		self.labels = tf.placeholder(shape=[num_labels,None], dtype=tf.float32, name="labels")
		self.embeddings = tf.placeholder(shape=[vocabulary_size,embedding_size], dtype=tf.float32,name="embeddings")
		self.embed = tf.nn.embedding_lookup(self.embeddings, self.inp_x)
		self.global_step = tf.Variable(0, dtype=tf.int32)
		self.classifiers=[]
		classify_summaries=[]
		for i in range(num_labels):
			with tf.variable_scope('label_classifier-'+str(i)):
				csfr = classifier_core_layer(embedding_size, hidden_size, self.embed, self.labels[i], 0.01)					
				self.classifiers.append(csfr)
				classify_summaries.append(csfr.loss_summary)
				classify_summaries.append(csfr.l2_summary)
		self.loss = tf.reduce_mean([x.loss for x in self.classifiers])
		self.average_loss_summary = tf.summary.scalar("avg_loss",self.loss)
		self.all_summaries = tf.summary.merge(classify_summaries + [self.average_loss_summary])

def get_accuracy(pred, labels):
	acc=0
	n = 0
	c=0
	# not looping on labels because this makes testing easy.
	for i in range(len(pred)):
		# c = 0
		for j in range(len(pred[0])):
			if labels[i][j]==1:
				n+=1
				if pred[i][j]==labels[i][j]:
					c+=1
		# acc+= float(c) / len(pred[0])
	if n==c:
		return 'N/A'
	return 100 * c/ max(n,1)#acc / len(pred)

def createInpOutListsFromBatch(batch):
	inp_x = []
	inp_y=[]
	# separate batch into inp and op
	for k in batch:
		inp_x.append(k[0])
		inp_y.append(k[1])
	return inp_x, inp_y

def executeTraining(train_dataset_merged, valid_dataset_merged, num_epochs, batch_size, vocabulary_size, embedding_size, num_labels, hidden_size, 
					summary_frequency, embeddings_file, log_directory):
	graph = tf.Graph()
	with graph.as_default():
		tg = TrainingGraph(vocabulary_size, embedding_size, num_labels, hidden_size)
	train_batch = BatchGenerator(train_dataset_merged, batch_size, num_labels)
	valid_batch = BatchGenerator(valid_dataset_merged, len(valid_dataset_merged), num_labels)

	train_log_directory = os.path.join(log_directory,"train")
	train_summary_directory = os.path.join(train_log_directory,"summaries")
	train_model_directory = os.path.join(train_log_directory,"models")
	valid_log_directory = os.path.join(log_directory, "valid")
	# utility.makeDir(train_summary_directory)
	# utility.makeDir(train_model_directory)
	# utility.makeDir(valid_log_directory)

	num_iters = (len(train_dataset_merged) // batch_size) * num_epochs
	feed_dict={}
	embeddings = utility.loadEmbeddings(embeddings_file)
	feed_dict[tg.embeddings] = embeddings
	print("Will take {} iters".format(num_iters))
	# average loss of the system as whole
	overall_avg_loss = 0.0
	with tf.Session(graph=graph) as session:
		session.run(tf.global_variables_initializer())
		for i in range(num_iters):
			batch = train_batch.next_batch()
			feed_dict[tg.inp_x], lbls_batch = createInpOutListsFromBatch(batch)
			feed_dict[tg.labels] = np.transpose(lbls_batch)
			chk = False
			if 1 in feed_dict[tg.labels][0]:
				chk=True
			# print(feed_dict[tg.labels])
			# dec=[hotDecode(x) for x in feed_dict[tg.labels]]
			# print(dec)
			# lbls = np.array(feed_dict[tg.labels])
			# print(lbls[:,dec[0][0]-1])
			# break
			# break

			# store loss and predictions for each classifier. Loss is for general overall loss average
			# calculation of system as a whole, storing labels to calculate accuracy.
			classifier_ops = []
			# loss across all classifiers.
			net_loss = 0.0
			for j in range(10):
				cl, prediction, _ = session.run([tg.classifiers[j].loss, tg.classifiers[j].prediction, tg.classifiers[j].optimizer], feed_dict=feed_dict)
				print(prediction)
				classifier_ops.append(prediction)
			# for x in range(len(prediction)):
			# 	print(prediction[i],"--",feed_dict[tg.labels][i])
			# print(cl)
			# break
			net_loss+=cl
			# print(prediction)
			# average of loss across all classifiers
			# net_loss/=len(tg.classifiers)
			# print(net_loss)
			# classifier_ops[0][4] = [-1]
			# break
			overall_avg_loss+=net_loss
			# classifier_ops is num_classifiers x batch_size.
			# convert it to batch x num_classifiers
			classifier_ops = np.transpose(classifier_ops,[1,0,2])
			# print(classifier_ops)
			# storing one hot vecs in diff var to get rid of third dimension while generating one hot vecs.
			# sort of concatenate for free at cost for memory 
			pred_hot_vec = []
			for j in range(len(classifier_ops)):
				pred_hot_vec.append(hotEncodeDistribution(classifier_ops[j]))
			# memory cleanup, bit agressive
			del classifier_ops
			# print(pred_hot_vec)
			# print("labels")
			# print(feed_dict[tg.labels])
			# break
			acc = get_accuracy(pred_hot_vec, lbls_batch)
			print("step {}/{}: loss: {}, accuracy:{}".format(i,num_iters,net_loss, acc))
			if i==summary_frequency:
				print((len(pred_hot_vec),len(pred_hot_vec[0])), (len(lbls_batch),len(lbls_batch[0])))
				for j in range(len(pred_hot_vec)):
					print(pred_hot_vec[j],"--",lbls_batch[j])
					print()
				# break
			# break

parser = argparse.ArgumentParser()
parser.add_argument("--embedding-file",help="embeddings txt file to read from", required=True)
args = parser.parse_args()

res = loadDataset("../data/nodes.csv","../data/groups.csv","../data/group-edges.csv")
dataset = res['node2labels']
nodes=res['nodes']
labels = res['labels']
split_ratio = 0.75
random.shuffle(dataset)
splitBorder = int(len(dataset)*split_ratio)
train_dataset = dataset[:splitBorder]
valid_dataset = dataset[splitBorder:]
train_dataset_merged = collectNodesAndLabels(train_dataset)
valid_dataset_merged = collectNodesAndLabels(valid_dataset)			
# print(batch2string(train_batch.next_batch()))
# t = TrainingGraph(len(nodes), 128, len(labels), 15)
executeTraining(train_dataset_merged, valid_dataset_merged, 1, 5, len(nodes), 128, len(labels), 15, 
					10, args.embedding_file, "memes")
log_directory = "directory from loading embeddings " + "classifier_runs" + "timestamp"
write_metadata = "in metadata.txt, which embeddings file this thing trained and ran on"
# print(t.classifiers[0].loss)