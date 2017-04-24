import onevAll as classifier
import numpy as np
import tensorflow as tf
import argparse
import utility
import os
import time
import sys
import ast

def readLabelsFile(filename, delim=","):
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

def hotDecode(prediction):
	p=[]
	for i in range(len(prediction)):
		if prediction[i]==1:
			p.append(classifier.id2label(i))
	return p

def readFile(filename):
	line=""
	with open(filename,'r') as f:
		line = f.read()
	return line

"""
From stdin reads a node name on each line. Converts it to node2id representation used in previous 
training steps and tries to predict a list of classifiers.
Can speed up execution by giving a comma separated list of nodes (batch_size amount of nodes).
In case of evaluation, provide the cmd line parameters realted to 'eval-file'
"""
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# parser.add_argument("--input-file",help="Input file for evaluating the model", required=True)
	parser.add_argument("--model-directory",help="Directory that contains the model", required=True)
	parser.add_argument("--eval-file",help="File that contains list of labels and nodes (used for evaluating the model). <Node,Label> entires expected", default=None)
	parser.add_argument("--eval-file-delim",help="Delimiter for the node-label file.", default=",")
	parser.add_argument("--log-dir",help="Main directory to store summaries (used for evaluating the model)", default="eval_runs")
	args = parser.parse_args()
	configfile = os.path.join(args.model_directory,'metadata.txt')
	config_dict = ast.literal_eval(readFile(configfile))
	config = utility.ConfigProvider()
	config.setDict(config_dict)
	vocabulary_size = config.getOption('num_nodes')
	num_labels = config.getOption('num_labels')
	hidden_size = config.getOption('hidden_size')
	embedding_file_location = config.getOption('embedding_file')
	
	configfile = os.path.join(args.model_directory,'classify_config.txt')
	config_dict = ast.literal_eval(readFile(configfile))
	config = utility.ConfigProvider()
	config.setDict(config_dict)
	embedding_size = config.getOption('embedding_size')
	
	model_directory = os.path.join(args.model_directory,'models')
	timestamp = int(time.time())
	log_dir = os.path.join(args.log_dir,str(timestamp))
	eval_log_dir = os.path.join(log_dir,'eval_summaries')
	nodeToLabel = None
	
	if args.eval_file!=None:
		nodeToLabel = readLabelsFile(args.eval_file, args.eval_file_delim)
		utility.makeDir(eval_log_dir)
		meta_file_path = os.path.join(log_dir,"meta.txt")
		classifier.writeMeta(meta_file_path,num_nodes = vocabulary_size, num_labels=num_labels,hidden_size=hidden_size,
							 embedding_size=embedding_size,learned_from=args.model_directory,eval_file=args.eval_file)
	
	graph = tf.Graph()
	with graph.as_default():
		tg = classifier.TrainingGraph(vocabulary_size, embedding_size, num_labels, hidden_size)
		f1_tf = tf.placeholder(shape=[],name='f1',dtype=tf.float32)
		prec_tf = tf.placeholder(shape=[],name='precision',dtype=tf.float32)
		rec_tf =tf.placeholder(shape=[],name='recall',dtype=tf.float32)
		avg_f1_tf = tf.placeholder(shape=[],name='avg_f1',dtype=tf.float32)
		avg_prec_tf = tf.placeholder(shape=[],name='avg_prec',dtype=tf.float32)
		avg_rec_tf = tf.placeholder(shape=[],name='avg_rec',dtype=tf.float32)

		f1_summary = tf.summary.scalar("f1_summary",f1_tf)
		precision_summary = tf.summary.scalar("precision_summary",prec_tf)
		recall_summary = tf.summary.scalar("recall_summary",rec_tf)
		stat_summary = tf.summary.merge([f1_summary, precision_summary, recall_summary])
		avg_f1_summary = tf.summary.scalar("avg_f1_summary",avg_f1_tf)
		avg_precision_summary = tf.summary.scalar("avg_precision_summary",avg_prec_tf)
		avg_recall_summary = tf.summary.scalar("avg_recall_summary",avg_rec_tf)
		stat_dict={}
		feed_dict={}

	summary_writer=None
	if nodeToLabel!=None:
		summary_writer = tf.summary.FileWriter(eval_log_dir,graph=graph)

	with tf.Session(graph=graph) as session:
		saver = tf.train.Saver(tf.global_variables())
		ckpt = tf.train.get_checkpoint_state(model_directory)
		saver.restore(session, ckpt.model_checkpoint_path)
		embeddings = utility.loadEmbeddings(embedding_file_location)
		feed_dict[tg.embeddings] = embeddings
		global_count = 0
		for line in sys.stdin.readlines():
			if len(line.strip("\n "))==0:
				continue
			line = [int(x) for x in line.split(",")]
			inp = [classifier.node2id(x) for x in line]
			preds=[]
			labels=[]	# populated in case of evaluation, left empty otherwise.
			labelsForCalc=[]
			for j in range(num_labels):
				lblArr=[]
				feed_dict[tg.classifiers[j].inp_x] = inp
				if nodeToLabel==None:
					prediction = session.run([tg.classifiers[j].prediction], feed_dict=feed_dict)
					prediction = prediction[0]
				else:
					lbls = [[1,0] if j in nodeToLabel[x] else [0,1] for x in inp]
					lblArr.append(lbls)	# for running stats across whole batch
					labelsForCalc.append([x[0] for x in lbls])
					feed_dict[tg.classifiers[j].labels] = lbls
					loss, loss_summary, prediction = session.run([tg.classifiers[j].loss,tg.classifiers[j].loss_summary,tg.classifiers[j].prediction], feed_dict=feed_dict)
				# print(prediction)
				preds.append(prediction)
				labels.append({'labels':lblArr})
			preds_hot_vec = []
			for j in range(len(preds)):
				preds_hot_vec.append(classifier.hotEncodeDistribution(preds[j]))

			lbl_transposed=None
			if nodeToLabel!=None:
				f1,prec,rec = classifier.get_accuracy(preds_hot_vec,labels)
				stat_dict[f1_tf] = f1
				stat_dict[prec_tf] = prec
				stat_dict[rec_tf] = rec
				f1_summ, prec_summ, rec_summ, stat_summ = session.run([f1_summary, precision_summary, recall_summary, stat_summary], feed_dict = stat_dict)
				summary_writer.add_summary(stat_summ, global_count)
				lbl_transposed = np.transpose(labelsForCalc)
				print("f1:{}, prec:{}, rec:{}".format(f1,prec,rec))
			# convert [classifiers x batch] to [batch x classifers]
			preds_hot_vec = np.transpose(preds_hot_vec)
			for i in range(len(inp)):
				lbls = sorted(hotDecode(preds_hot_vec[i]))
				lbls = " ".join([str(x) for x in lbls])
				nodeid = classifier.id2node(inp[i])
				if nodeToLabel==None:
					print("{}:{}".format(nodeid, lbls))
				else:
					exp_lbls = sorted(hotDecode(lbl_transposed[i]))
					exp_lbls = " ".join([str(x) for x in exp_lbls])
					print("{}: expected: {} got: {}".format(nodeid,exp_lbls,lbls))
			global_count+=1
			summary_writer.flush()






