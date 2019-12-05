import pandas as p
import random
from sklearn import metrics

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
		self.testLabel = []
		self.predictedLabel = []

	def getNextSet(self):
		test = self.df.loc[self.listCV[self.currentFold],:]
		trainIndex = list(set(self.originalIndex).difference(set(self.listCV[self.currentFold])))
		train = self.df.loc[trainIndex,:]
		self.currentFold = (self.currentFold+1)%self.kfolds
		return train.reset_index(),test.reset_index()

	def addResults(self,labelActual,labelPredicted):
		labelActual = list(labelActual.values)
		self.testLabel = [*self.testLabel,*labelActual]
		self.predictedLabel = [*self.predictedLabel,*labelPredicted]

	#Regression
	def computeScoreRegression(self):
		#r = metrics.explained_variance_score(self.testLabel,self.predictedLabel)
		r = metrics.mean_squared_error(self.testLabel,self.predictedLabel)
		#r = metrics.r2_score(self.testLabel,self.predictedLabel)
		return r

	def runTestRegression(self,machine,labelColumn):
		for i in range(self.kfolds):
			train, test = self.getNextSet()
			machine.fit(train.drop(columns=[labelColumn]), train[labelColumn])
			predicted = machine.predict(test.drop(columns=[labelColumn]))
			self.addResults(test[labelColumn],predicted)
		return self.computeScoreRegression()

	#Classification
	def computeScoreClassification(self):
		r = metrics.accuracy_score(self.testLabel,self.predictedLabel)
		#r = metrics.cohen_kappa_score(self.testLabel,self.predictedLabel)
		#r = metrics.confusion_matrix(self.testLabel,self.predictedLabel)
		#r = metrics.f1_score(self.testLabel,self.predictedLabel)
		#r = metrics.auc(self.testLabel,self.predictedLabel)
		return r

	def runTestClassification(self,machine,labelColumn):
		for i in range(self.kfolds):
			train, test = self.getNextSet()
			machine.fit(train.drop(columns=[labelColumn]), train[labelColumn])
			predicted = machine.predict(test.drop(columns=[labelColumn]))
			self.addResults(test[labelColumn],predicted)
		return self.computeScoreClassification()
