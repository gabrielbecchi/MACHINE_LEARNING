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
		self.meanSquaredError = []

	def getNextSet(self):
		test = self.df.loc[self.listCV[self.currentFold],:]
		trainIndex = list(set(self.originalIndex).difference(set(self.listCV[self.currentFold])))
		train = self.df.loc[trainIndex,:]
		self.currentFold = (self.currentFold+1)%self.kfolds
		return train.reset_index(),test.reset_index()

	def addResults(self,labelActual,labelPredicted):
		labelActual = list(labelActual.values)
		sumSquares = 0
		for i in range(len(labelActual)):
			#print(str(labelActual[i])+" - "+str(labelPredicted[i]))
			sumSquares += (labelActual[i]-labelPredicted[i])**2
		self.meanSquaredError.append((sumSquares/len(labelActual))**0.5)
		#input()

	def rootMeanSquaredError(self):
		score = (sum(self.meanSquaredError)/len(self.meanSquaredError))
		self.meanSquaredError = []
		return score

	def runTest(self,machine,labelColumn):
		for i in range(self.kfolds):
			train, test = self.getNextSet()
			machine.fit(train.drop(columns=[labelColumn]), train[labelColumn])
			predicted = machine.predict(test.drop(columns=[labelColumn]))
			self.addResults(test["price"],predicted)
		return self.rootMeanSquaredError()
