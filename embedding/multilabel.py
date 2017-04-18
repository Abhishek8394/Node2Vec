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
			node2labels.append((entry[0], entry[1]))
	reverseNodes = dict(zip(nodes.values(), nodes.keys()))
	reverseLabels = dict(zip(labels.values(), labels.keys()))
	return {"nodes":nodes, "labels":labels, "node2labels":node2labels, "reverseNodes":reverseNodes, "reverseLabels":reverseLabels}

def writeMeta(meta_file, embedding_file, hidden_size):
	j = {'embedding_file':embedding_file, 'hidden_size':hidden_size}
	with open(meta_file,"w") as f:
		f.write(str(j))

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
	# print(probabilities, sum(probabilities))
	for i in probabilities:
		p.append(1.0 if i>=0.50 else 0.0)
	return p 

def hotEncode(num_labels, labels):
	hotVec = [0] * num_labels
	for i in labels:
		hotVec[label2id(i)] = 1
	return hotVec

def hotDecode(hotVec):
	s = [id2label(i) for i in range(len(hotVec)) if hotVec[i]==1]
	return s

"""
Each row of a batch will be : [node_id,[encoding of labels]]
encoding of each label = [1,0] if node has label else [0,1]
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
		rec_labels = self.dataset[node]
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
		# first output is yes, second is no.
		self.w2 = tf.Variable(tf.truncated_normal([hidden_size,2],-0.1,0.1), dtype=tf.float32, name="weight2")
		self.b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name="bias2")
		self.logits = tf.matmul(self.o1, self.w2) + self.b2
		# print(self.o1.shape,self.logits.shape)
		# return
		# labels = tf.reshape(labels,shape=[-1,2])
		self.prediction = tf.nn.softmax(self.logits, name="prediction")
		self.l2_loss = tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels)) + (l2_lambda * self.l2_loss)
		# loss = tf.reduce_mean(-(labels * tf.log(self.prediction + 1e-7) + (1-labels)*tf.log(1-self.prediction + 1e-7))) + (l2_lambda * self.l2_loss) 
		# self.loss = tf.identity(loss, name="loss")
		self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
		self.l2_summary = tf.summary.scalar("l2_summary", self.l2_loss)
		self.loss_summary = tf.summary.scalar("loss_summary", self.loss)

class TrainingGraph(object):
	def __init__(self, vocabulary_size, embedding_size, num_labels, hidden_size):
		self.inp_x = tf.placeholder(shape=[None], dtype=tf.int32, name="inp_x")
		self.labels = tf.placeholder(shape=[None,num_labels], dtype=tf.float32, name="labels")
		self.embeddings = tf.placeholder(shape=[vocabulary_size,embedding_size], dtype=tf.float32,name="embeddings")
		self.embed = tf.nn.embedding_lookup(self.embeddings, self.inp_x)
		self.global_step = tf.Variable(0, dtype=tf.int32)
		self.w1 = tf.Variable(tf.random_normal([embedding_size, hidden_size],-1,1), dtype=tf.float32, name="weight1")
		self.b1 = tf.Variable(tf.zeros([hidden_size]), dtype=tf.float32, name="bias1")
		self.o1 = (tf.matmul(self.embed,self.w1) + self.b1)
		# first output is yes, second is no.
		self.w2 = tf.Variable(tf.random_normal([hidden_size,num_labels],-1,1), dtype=tf.float32, name="weight2")
		self.b2 = tf.Variable(tf.zeros([num_labels]), dtype=tf.float32, name="bias2")
		self.logits = tf.matmul(self.o1, self.w2) + self.b2
		self.prediction = tf.sigmoid(self.logits, name='prediction')
		self.l2_loss = tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2)
		self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = self.logits, targets = self.labels, pos_weight=5))# + (0.01 * self.l2_loss)
		# self.loss = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(self.prediction + 1e-8),1))# + (0.01 * self.l2_loss)
		self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
		self.l2_summary = tf.summary.scalar("l2_summary", self.l2_loss)
		self.loss_summary = tf.summary.scalar("loss_summary", self.loss)
		self.all_summaries = tf.summary.merge([self.loss_summary, self.l2_summary])


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
			if labels[i][j]==1:
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
	valid_batch = BatchGenerator(valid_dataset_merged, len(valid_dataset_merged), num_labels)
	
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
	
	num_iters = (len(train_dataset_merged) // batch_size) * num_epochs
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
			feed_dict[tg.inp_x], lbls_batch = createInpOutListsFromBatch(batch)
			feed_dict[tg.labels] = lbls_batch #np.transpose(lbls_batch,[1,0])
			net_loss = 0.0
			loss, prediction, _, train_summary = session.run([tg.loss, tg.prediction, tg.optimizer, tg.all_summaries], feed_dict = feed_dict)
			net_loss+=loss
			overall_avg_loss+=net_loss

			pred_hot_vec = []
			for j in range(len(prediction)):
				pred_hot_vec.append(hotEncodeDistribution(prediction[j]))
			
			train_summary_writer.add_summary(train_summary,i)
			
			f1, precision, recall = get_accuracy(pred_hot_vec, lbls_batch)
			stat_dict[precision_tf] = precision
			stat_dict[recall_tf] = recall
			stat_dict[f1_tf] = f1
			pre, rec, ef1, ss = session.run([precision_tf, recall_tf, f1_tf, stat_summary],feed_dict = stat_dict)
			train_summary_writer.add_summary(ss,i)

			print("step {}/{}: loss: {}, f1:{}".format(i,num_iters,net_loss, f1))
			if i%summary_frequency==0:
				save_loc = saver.save(session, train_model_file, global_step = i)
				print("Saving model at {}".format(save_loc))
				print((len(pred_hot_vec),len(pred_hot_vec[0])), (len(lbls_batch),len(lbls_batch[0])))
				for j in range(len(pred_hot_vec)):
					print([str(x) for x in zip(pred_hot_vec[j],lbls_batch[j])])
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
num_epochs = 3			
batch_size = 5
hidden_size = 150
embedding_size = 128
summary_frequency = 10
timestamp = int(time.time())
log_directory = os.path.join("classifier_runs",str(timestamp))
utility.makeDir(log_directory)
write_metadata = os.path.join(log_directory,"metadata.txt")
writeMeta(write_metadata, args.embedding_file, hidden_size)
executeTraining(train_dataset_merged, valid_dataset_merged, num_epochs, batch_size, len(nodes), embedding_size, len(labels), hidden_size, 
					summary_frequency, args.embedding_file, log_directory)
