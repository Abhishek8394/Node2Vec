import tensorflow as tf
import random
"""
nodefile -> just a list of all possible nodes.
labelfile -> just a list of all possible labels.
node2labelFile -> label assignments for node. 
"""

# Since all nodes and labels are 1..n and we need 0..n-1
def node2id(node):
	return node-1

def id2node(_id):
	return _id+1

def label2id(node):
	return node-1

def id2label(_id):
	return _id+1

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
		p = 1.0 if i>=0.5 else 0.0
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
		self.w1 = tf.Variable(tf.truncated_normal([embedding_size, hidden_size]), dtype=tf.float32, name="weight1")
		self.b1 = tf.Variable(tf.zeros([hidden_size]), dtype=tf.float32, name="bias1")
		self.o1 = tf.sigmoid(tf.matmul(embed,self.w1) + self.b1)
		self.w2 = tf.Variable(tf.truncated_normal([hidden_size,1]), dtype=tf.float32, name="weight2")
		self.b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name="bias2")
		self.logits = tf.matmul(self.o1, self.w2) + self.b2
		# self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels[i]))
		self.prediction = tf.nn.softmax(self.logits, name="prediction")
		self.l2_loss = tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2)
		loss = tf.reduce_mean(-(labels * tf.log(self.prediction) + (1-labels)*tf.log(1-self.prediction))) + (l2_lambda * self.l2_loss) 
		self.loss = tf.identity(loss, name="loss")
		self.optimizer = tf.train.AdadeltaOptimizer(1e-3).minimize(self.loss)
		self.l2_summary = tf.summary.scalar("l2_summary", self.l2_loss)
		self.loss_summary = tf.summary.scalar("loss_summary", self.loss)

class TrainingGraph(object):
	def __init__(self, vocabulary_size, embedding_size, num_labels, hidden_size):
		self.inp_x = tf.placeholder(shape=[None], dtype=tf.int32, name="inp_x")
		self.labels = tf.placeholder(shape=[None,num_labels], dtype=tf.float32, name="labels")
		self.embeddings = tf.placeholder(shape=[vocabulary_size,embedding_size], dtype=tf.float32,name="embeddings")
		self.embed = tf.nn.embedding_lookup(self.embeddings, self.inp_x)
		self.global_step = tf.Variable(0, dtype=tf.int32)
		self.classifiers=[]
		for i in range(num_labels):
			with tf.variable_scope('label_classifier-'+str(i)):
				csfr = classifier_core_layer(embedding_size, hidden_size, self.embed, self.labels[i], 0.01)	
				self.classifiers.append(csfr)
		self.loss = tf.reduce_mean([x.loss for x in self.classifiers])
		self.average_loss_summary = tf.summary.scalar("avg_loss",self.loss)


res = loadDataset("../data/nodes.csv","../data/groups.csv","../data/group-edges.csv")
dataset = res['node2labels']
split_ratio = 0.75
random.shuffle(dataset)
splitBorder = int(len(dataset)*split_ratio)
train_dataset = dataset[:splitBorder]
valid_dataset = dataset[splitBorder:]
train_dataset_merged = collectNodesAndLabels(train_dataset)
valid_dataset_merged = collectNodesAndLabels(valid_dataset)

def executeTraining(train_dataset_merged, valid_dataset_merged, num_epochs, batch_size, vocabulary_size, embedding_size, num_labels, hidden_size, 
					summary_frequency):
	graph = tf.Graph()
	with graph.as_default():
		tg = TrainingGraph(vocabulary_size, embedding_size, num_labels, hidden_size)
	train_batches = BatchGenerator(train_dataset_merged, batch_size, num_labels)
	valid_batches = BatchGenerator(valid_dataset_merged, len(valid_dataset_merged), num_labels)

	num_iters = (len(train_dataset_merged) // batch_size) * num_epochs
	print("Will take {} iters".format(num_iters))
	with tf.Session(graph=graph) as session:
		for i in range(num_iters):
			batch = train_batch.next_batch()
			inp_x = []
			inp_y=[]
			# separate batch into inp and op
			for k in batch:
				inp_x.append(k[0])
				inp_y.append(k[1])
			
# print(batch2string(train_batch.next_batch()))
t = TrainingGraph(10000, 128, 40, 15)
# print(t.classifiers[0].loss)