import numpy as np
import random
import argparse

def loadDataset(fileName, dataDelim=":", max_rec = -1):
	ds = []#np.empty([0,None],dtype=np.int32)
	nodes=set()
	ctr = 0
	s = None
	with open(fileName,"r") as f:
		for line in f:
			# print(ctr)
			if max_rec!=-1 and ctr>=max_rec:
				break
			l = line.strip("\n "+dataDelim).split(dataDelim)
			# walkObj={}
			# walkObj['node'] = int(l[0])
			walkObj = [int(x) for x in l if x!='']
			ds.append(walkObj)
			nodes.add(walkObj[0])
			ctr+=1
	num_nodes = len(nodes)
	return {"dataset": ds, "num_nodes": num_nodes}

def readLabelsFlle(filename, delim=","):
	nodeToLabel = {}
	with open(filename,"r") as f:
		for j in f:
			line = [int(x) for x in j.strip("\n").split(delim)]
			node = classifier.node2id(line[0])
			label = classifier.label2id(line[1])
			if node not in nodeToLabel:
				nodeToLabel[node] = set()
			nodeToLabel[node].add(label)
	return nodeToLabel

"""
since graphs contain vertices 1...n and we need to map them to 0 - (vocab_size-1)
switch these implementations for differnet graph
"""
def node2id(node):
	return node-1

def id2node(_id):
	return _id+1

def label2id(node):
	return node-1

def id2label(_id):
	return _id+1

class BatchGenerator(object):
	"""
		if you want to capture t-k, t-k+1,...,t-1,t+1,...t+k pairs, set window_size to k.
		window_size > 0.
		batch_size = number of walks to sample from. 
		total batch size of single batch = batch_size * (2 * window_size)
	"""
	def __init__(self, dataset, batch_size, window_size):
		self.ds = dataset
		self.batch_size = batch_size
		self.window_size = window_size
		num_partitions = len(dataset) // batch_size
		self.cursors = [i*batch_size for i in range(num_partitions)]
		self.rec_cursors = [0] * len(self.cursors)

	def getResultantBatchSize(self):
		return self.batch_size * 2 * self.window_size

	def getWindowBatch(self, cur_ind):
		nds = self.ds[self.cursors[cur_ind]] # list of nodes as a walk
		prevInd = max(0,self.rec_cursors[cur_ind] - self.window_size)
		maxNxt = min(self.rec_cursors[cur_ind] + self.window_size, len(nds)-1)
		batch=[]
		labels=[]
		# TODO instead of 50%of database, take 50% of each walk. Also maybe invert batch and label pairs too?
		# modified version, node vs walks. (Not how the paper did)
		# if len(nds)==1:
		# 	for i in range(window_size):
		# 		batch.append(node2id(nds[0]))
		# 		labels.append([node2id(nds[0])])
		# else:
		# 	for i in range(1,len(nds)):
		# 		batch.append(node2id(nds[0]))
		# 		labels.append([node2id(nds[i])])
		# self.cursors[cur_ind] = (self.cursors[cur_ind] + 1) % len(self.ds)
		# return {"batch": batch, "label":labels} 

		# TODO-> run with following model.
		# literal skip gram.
		for i in range(prevInd, maxNxt+1, 1):
			if i==self.rec_cursors[cur_ind]:
				continue
			cur_word = self.ds[self.cursors[cur_ind]][self.rec_cursors[cur_ind]]
			batch.append(node2id(cur_word))
			# since we predict context from given word, context word is our label
			cur_label = self.ds[self.cursors[cur_ind]][i]
			labels.append([node2id(cur_label)])

		self.rec_cursors[cur_ind] += 1

		# if all nodes of current record processed, go to next record (walk).
		if self.rec_cursors[cur_ind] == len(nds):
			self.rec_cursors[cur_ind] = 0
			self.cursors[cur_ind] = (self.cursors[cur_ind] + 1) % len(self.ds)
		return {"batch": batch, "label":labels} 

	def next_batch(self):
		batch=[]
		label=[]
		for i in range(self.batch_size):
			b = self.getWindowBatch(i)
			batch.extend(b['batch'])
			label.extend(b['label'])

		return {'batch': batch, 'label': label}

def batch2string(batch, label = None):
	s=[]
	if label==None:
		for i in range(len(batch)):
			s.append(str(id2node(batch[i])))
	else:
		for i in range(len(batch)):
			s.append(str(id2node(batch[i])) + ":" + str(id2node(label[i])))
	return ", ".join(s)


"""
For creating balanced amount of positive and negative samples for a classifier.
Output is no format: label, node, 1 if node has label
					 label, node, 0 if node doesn't have label
"""
def createBalancedDataset(node2labelsFile, neg2posRatio, outputFile, dataDelim=','):
	nodes = set()
	label2nodes={}
	with open(node2labelsFile,'r') as f:
		for line in f:
			nums = line.strip('\n').split(dataDelim)
			n = int(nums[0])
			l = int(nums[1])
			if l not in label2nodes:
				# using list instead of set to accommodate multiple copies in case required by dataset
				label2nodes[l]=[]
			label2nodes[l].append(n)
			nodes.add(n)

	with open(outputFile,'w') as f:
		for i in label2nodes:
			posSet = set(label2nodes[i])
			unsetNodes = [x for x in nodes if x not in posSet]
			maxNumNeg = int(neg2posRatio * len(posSet))
			random.shuffle(unsetNodes)
			negNodes = unsetNodes[:maxNumNeg]
			print('Label {}: pos:{} neg:{}'.format(i,len(label2nodes[i]),len(negNodes)))
			for j in label2nodes[i]:
				f.write("{},{},1\n".format(i,j))
			for j in negNodes:
				f.write("{},{},0\n".format(i,j))

def trainSingleClassifier_old_v2(classifierId, graph, session, trainingGraph, dataset, batch_size, embeddings, batchGen, num_epochs, 
						  train_summary_writer, saver, train_model_file, is_training, valid_test, summary_frequency=-1):
	tg = trainingGraph
	with graph.as_default():
		precision_tf = tf.placeholder(shape=[], dtype=tf.float32,name='precision')
		recall_tf = tf.placeholder(shape=[], dtype=tf.float32,name='recall')
		f1_tf = tf.placeholder(shape=[], dtype=tf.float32,name='f1')
		with tf.variable_scope('label_classifier-'+str(classifierId)):
			macro_f1_tf = tf.placeholder(shape=[], dtype=tf.float32,name='macro_f1')
	total_prec = 0.0
	total_rec = 0.0
	precision_summary = tf.summary.scalar('precision_summary',precision_tf)
	recall_summary = tf.summary.scalar('recall_summary',recall_tf)
	f1_summary = tf.summary.scalar('f1_summary',f1_tf)
	macro_f1_summary = tf.summary.scalar('macro_f1_summary', macro_f1_tf)
	stat_summary = tf.summary.merge([precision_summary, recall_summary, f1_summary])
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

		f1,prec,rec = get_accuracy(pred, [batch])
		total_rec+=rec
		total_prec+=prec
		print("step: {} loss:{} f1:{}".format(i,net_loss,f1))
		summaries = op['summaries_calculated']
		for j in range(len(summaries)):
			train_summary_writer.add_summary(summaries[j], i)
		if summary_frequency > 0 and i%summary_frequency == 0:
			save_loc = saver.save(session, train_model_file, global_step=i)
			valid_test.run_validation(session, tg, 1, feed_dict, [classifierId])
			print(pred,batch['labels'])
	valid_test.run_validation(session, tg, 1, feed_dict, [classifierId])
	avg_prec = total_prec/num_iters
	avg_rec = total_rec/num_iters
	macro_f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec + 1e-7)
	macro_f1_summ = session.run([macro_f1_summary], feed_dict={macro_f1_tf:macro_f1})
	train_summary_writer.add_summary(macro_f1_summ[0],0)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--bal-data',help='run the dataset balancer',action = 'store_true')
	parser.add_argument('--data-file', help='dataset file to balance', default="../data/group-edges.csv")
	parser.add_argument('--out-file', help='destination to store output of balanced dataset',default='../data/balanced-group-edges.csv')
	parser.add_argument('--neg-ratio',help = 'neg to pos ratio of samples desired', type=float, default = 2)
	parser.add_argument('--data-delim',help = 'Delimeter used in dataset file', default = ',')
	args = parser.parse_args()
	if args.bal_data:
		createBalancedDataset(args.data_file, args.neg_ratio, args.out_file, args.data_delim)
