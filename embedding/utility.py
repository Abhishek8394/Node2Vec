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