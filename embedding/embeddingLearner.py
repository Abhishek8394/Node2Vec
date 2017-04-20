import sys
import argparse
import os
import time
import dataHandler as dh
import utility
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

def createLogDirectories(log_dir):
	timestamp = int(time.time())
	main_dir = os.path.join(log_dir, str(timestamp))
	train_log_dir = os.path.join(main_dir,"train")
	checkpoint_dir = os.path.join(main_dir,"checkpoints")
	utility.makeDir(train_log_dir)
	utility.makeDir(checkpoint_dir)
	return {"main_dir": main_dir, "train_log_dir": train_log_dir, "checkpoint_dir": checkpoint_dir}

def createTrainingGraph(vocabulary_size, embedding_size, nce_sample_size, learning_rate = 1.0):
	graph = tf.Graph()
	with graph.as_default():
		inp_x = tf.placeholder(shape=[None], dtype=tf.int32, name="inp_x")
		inp_y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name="inp_y") # "1" since nce_loss takes labels of size (batch, num_correct)
		embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size], -1.0, 1.0), dtype=tf.float32, name="embeddings")
		embed = tf.nn.embedding_lookup(embeddings, inp_x,name="embedding_lookup")
		w_nce = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], -0.1, 0.1), dtype=tf.float32, name="nce_w")
		b_nce = tf.Variable(tf.zeros([embedding_size]), dtype=tf.float32, name="nce_b")
		# logits = tf.nn.xw_plus_b(embed, w, b)
		loss = tf.reduce_mean(tf.nn.nce_loss(w_nce, b_nce, 
								labels=inp_y, inputs=embed, 
								num_sampled=nce_sample_size, 
								num_classes = vocabulary_size), name="loss")
		loss_summary = tf.summary.scalar("loss_summary", loss)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		grads =  optimizer.compute_gradients(loss)
		gradHist=[]
		for g,v in grads:
			if g is not None:
				gs = tf.summary.histogram("{}/grads/hist".format(v.name),g)
				gradHist.append(gs)
		gradSummary = tf.summary.merge(gradHist)
		optimize = optimizer.apply_gradients(grads)
		train_summary = tf.summary.merge([gradSummary, loss_summary])

	return {"graph":graph, "inp_x": inp_x, "inp_y":inp_y, "embeddings": embeddings, "loss":loss,
			"optimizer": optimizer, "optimize": optimize,"loss_summary":loss_summary, "gradSummary":gradSummary, 
			"train_summary": train_summary}

def executeTrainingGraph(graphVars, logdirs, dataset, batch_size, window_size, num_epochs, summary_frequency, num_checkpoints):
	graph = graphVars['graph']
	summary_directory = logdirs['train_log_dir']
	checkpoint_dir = logdirs['checkpoint_dir']
	
	walk_length = len(dataset[0])
	train_batches = dh.BatchGenerator(dataset, batch_size, window_size)
	actual_batch_size = train_batches.getResultantBatchSize()
	# since walk_length iters required for current batch of cursors to move to next record.
	# so we call walk length / batch size calls for processing entire dataset. Multiply num_epochs for several 
	# epochs on entire dataset.
	num_iters = (len(dataset) * num_epochs * walk_length) // batch_size
	print("Will run for {} iterations".format(num_iters))

	with tf.Session(graph=graph) as session:
		saver = tf.train.Saver(tf.global_variables(), max_to_keep = num_checkpoints)
		summaryWriter = tf.summary.FileWriter(summary_directory, graph=session.graph)
		session.run(tf.global_variables_initializer())
		net_loss = 0
		for i in range(num_iters):
			batch = train_batches.next_batch()
			feed_dict={}
			feed_dict[graphVars['inp_x']] = batch['batch']
			feed_dict[graphVars['inp_y']] = batch['label']
			loss, _ , train_summary= session.run([graphVars['loss'], graphVars['optimize'], graphVars['train_summary']], feed_dict = feed_dict)
			net_loss+=loss
			print("step {}/{}: loss: {}".format(i,num_iters,loss))
			summaryWriter.add_summary(train_summary,i)

			if i%summary_frequency==0:
				avg_loss = net_loss if i==0 else net_loss/i
				print("Average Loss: {}".format(avg_loss))
				path = saver.save(session, checkpoint_dir, global_step = i)
				print("Saved model at {}".format(path))
			
		# TODO better visualization
		# embedding_visualizer_config = projector.ProjectorConfig()
		# embedding = config.embeddings.add()
		# embedding.tensor_name = graphVars['embeddings'].name
		# embedding.metadata_path = os.path.join(summary_directory,'metadata.tsv')
		# projector.visualize_embeddings(summaryWriter, embedding_visualizer_config)

"""
output embeddings written as :
node_id:d1,d2,d3...
"""		
def printEmbeddings(graphVars, checkpoint_directory, print_only, output_file):
	graph = graphVars['graph']
	embeddings = res['embeddings']
	outputFilePath = os.path.join(checkpoint_directory,output_file)
	if not print_only:
		print("Will write embeddings in {}".format(outputFilePath))
		opfile = open(outputFilePath,"w")
	with tf.Session(graph=graph) as session:
		saver = tf.train.Saver(tf.global_variables())
		ckpt = tf.train.get_checkpoint_state(checkpoint_directory)
		saver.restore(session, ckpt.model_checkpoint_path)
		embed = session.run(embeddings)
		for i in range(len(embed)):
			opstring = str(i)+":"+",".join([str(x) for x in embed[i]])
			if print_only:
				print(opstring)
			else:
				opfile.write(opstring+"\n")
	if not print_only:
		opfile.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config-file", help="Config file for the training session", default="./train_config.txt")
	parser.add_argument("--input-file", help="Dataset of walks", required=False)
	parser.add_argument("--load-embed-from",help="just load pre-trained embeddings from a log directory", default=None)
	parser.add_argument("--print-only",help="only print the embeddings, do not write to file", action='store_true')
	parser.add_argument("--output-file", help="custom filename for storing embeddings.", default="embeddings.txt")
	parser.add_argument("--vocabulary-size", help="vocab size of embeddings.",type=int, default=10312)
	args = parser.parse_args()
	config = utility.ConfigProvider(args.config_file)	
	embedding_size = config.getOption('embedding_size')
	nce_sample_size = config.getOption('nce_sample_size')
	batch_size = config.getOption('batch_size')
	window_size = config.getOption('window_size')
	num_epochs = config.getOption('num_epochs')
	summary_frequency = config.getOption('summary_frequency')
	num_checkpoints = config.getOption('num_checkpoints')

	if args.load_embed_from!=None:
		res = createTrainingGraph(args.vocabulary_size, embedding_size, nce_sample_size)
		printEmbeddings(res, args.load_embed_from, args.print_only, args.output_file)
		exit(1)

	print("Loading dataset")
	ds_res = dh.loadDataset(args.input_file, max_rec=-1)
	data_split = config.getOption('data_split')
	dataset = ds_res['dataset']
	vocabulary_size = ds_res['num_nodes']
	ds_random_indices = np.arange(len(dataset))
	np.random.shuffle(ds_random_indices)
	splitBound = int(len(dataset)*data_split)
	ds_random_indices = ds_random_indices[:splitBound]
	train_dataset = []
	for i in ds_random_indices:
		train_dataset.append(dataset[i])
	print("Done.")

	print("Learning embeddings from {} % of dataset".format(data_split*100))
	print("Creating log dirs")
	logdirs = createLogDirectories(config.getOption('log_dir'))
	utility.copyFile(args.config_file,os.path.join(logdirs['main_dir'],'train_config.txt'))
	# logging on which walk dataset training was done.
	with open(os.path.join(logdirs['main_dir'],'meta.txt'),'w') as f:
		fileAddress={}
		fileAddress['cwd'] = os.getcwd()
		fileAddress['walksFile'] = args.input_file 
		f.write(str(fileAddress)+"\n")

	print("Done")
	# train_batches = dh.BatchGenerator(ds_res['dataset'], 5, 5)
	# print("Actual batch size: {}".format(train_batches.getResultantBatchSize()))
	# for i in range(5):
	# 	b = train_batches.next_batch()
	# 	print(dh.batch2string(b['batch'],b['label']))
	# 	print()
	res = createTrainingGraph(vocabulary_size, embedding_size, nce_sample_size)
	# res['log_directory'] = 'runs'
	executeTrainingGraph(res, logdirs, train_dataset, batch_size, window_size, num_epochs, summary_frequency, num_checkpoints)
	print("done")
