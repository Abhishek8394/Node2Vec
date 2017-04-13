import tensorflow as tf
import numpy as np


graph = tf.Graph()
with graph.as_default():
	inp_x = tf.placeholder(shape=[None,3],dtype=tf.float32)
	w1 = tf.constant([[1,0,0],[0,1,0],[0,0,1]],dtype=tf.float32)
	b1 = tf.constant([0,0,0],dtype=tf.float32)
	logit = tf.matmul(inp_x,w1) + b1
	sigmoid = tf.sigmoid(logit)
	# loss = tf.reduce_mean()

with tf.Session(graph=graph) as sess:
	ip = np.array([[1,2,3],[4,5,6]])
	fd={}
	fd[inp_x] = ip
	l,s = sess.run([logit,sigmoid],feed_dict=fd)
	ans=[]
	for i in range(len(ip)):
		log = np.dot(ip[i],[[1,0,0],[0,1,0],[0,0,1]])
		# log = np.sum(log,[10,10,10])
		ans.append(log)	
	print("l",l.shape)
	print("s",s.shape)
	for i in range(len(l)):
		print(l[i]," -- ",ans[i])
	print("sigs")
	print(s)