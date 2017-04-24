import os
import json

def makeDir(pth):
	if not os.path.exists(pth):
		os.makedirs(pth)

class ConfigProvider(object):
	def __init__(self, filename):
		self.js = None
		with open(filename) as f:
			self.js = json.load(f)

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
