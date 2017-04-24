import os
import json

def makeDir(pth):
	if not os.path.exists(pth):
		os.makedirs(pth)

class ConfigProvider(object):
	def __init__(self, filename=None):
		self.js = None
		if filename!=None:
			with open(filename) as f:
				self.js = json.load(f)
	def setDict(self,d):
		self.js = d

	def getOption(self,key):
		return self.js[key]

"""
Assuming all embeddings are stored by the embeddingTrainer. 
So will be in order of the embeddingsArray i.e. first line is index 0's embeddings and so on.
Stored embedding assumed to be in format: node_id:d1,d2,..
"""
def loadEmbeddings(embeddingsFile):
	embed=[]
	with open(embeddingsFile,"r") as f:
		for j in f:
			x = j.split(":")
			em = [float(x) for x in x[1].split(",")]
			embed.append(em)
	return embed

def copyFile(src, dest):
	with open(src,'r') as inp:
		with open(dest,'w') as out:
			for line in inp:
				out.write(line)

def trainSingleClassifier_one_vs_all_old(classifierId, graph, session, trainingGraph, dataset, batch_size, embeddings, batchGen, num_epochs,  train_summary_writer, saver, train_model_file, is_training, valid_test, summary_frequency=-1):
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