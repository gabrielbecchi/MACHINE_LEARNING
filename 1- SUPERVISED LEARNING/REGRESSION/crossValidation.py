import pandas as p
import random

class crossValidation:
	def __init__(self,df,kfolds):
		self.df = df
		self.kfolds = kfolds
		indexList = list(df.index.values)
		self.originalIndex = indexList
		self.listCV = []
		samplesPerFold = int(len(indexList)/kfolds)
		for i in range(kfolds-1):
			self.listCV.append(random.sample(indexList,samplesPerFold))
			indexList = list(set(indexList).difference(set(self.listCV[i])))
		self.listCV.append(indexList)
		self.currentFold = 0;

	def getNextSet(self):
		test = self.df.loc[self.listCV[self.currentFold],:]
		trainIndex = list(set(self.originalIndex).difference(set(self.listCV[self.currentFold])))
		train = self.df.loc[trainIndex,:]
		self.currentFold = (self.currentFold+1)%self.kfolds
		return train.reset_index(),test.reset_index()