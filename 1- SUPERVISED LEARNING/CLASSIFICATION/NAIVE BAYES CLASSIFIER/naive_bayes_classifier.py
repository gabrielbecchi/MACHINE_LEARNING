import pandas as p
import operator

class functionParameters:
	def __init__(self,label,xMean,xVar,yMean,yVar):
		self.label = label
		self.xMean = xMean
		self.xVar = xVar
		self.yMean = yMean
		self.yVar = yVar
	def getLabel(self):
		return self.label
	def getX(self):
		return xMean, xVar
	def getY(self):
		return yMean, yVar

	def calFuncValue(self,mean,var,value):
		return (-1*(((value-mean)**2)/var))
	def compValue(self,X,Y):
		return (self.calFuncValue(self.xMean,self.xVar,X)+self.calFuncValue(self.yMean,self.yVar,Y))


class naive_bayes_classifier:
	def train(self,train):
		self.listFunctions = []
		labels = train["Q"].unique()
		for label in labels:
			train_temp = train.loc[train["Q"] == label]
			xMean = train_temp["X"].mean()
			xVar = train_temp["X"].var() 
			yMean = train_temp["Y"].mean()
			yVar = train_temp["Y"].var()
			self.listFunctions.append(functionParameters(label,xMean,xVar,yMean,yVar))
		#print(self.listFunctions)

	def predict(self,x,y):
		dicPoints = {}
		for classType in self.listFunctions:
			dicPoints[classType.getLabel()] = classType.compValue(x,y)
		sorted_votes = sorted(dicPoints.items(), key=operator.itemgetter(1))
		#print(sorted_votes)
		return (sorted_votes[-1][0])

	def test(self,test):
		acc = 0
		acumulator = 0
		for index,serie in test.iterrows():
			acumulator += 1
			p = self.predict(serie["X"],serie["Y"])
			print(str(serie["X"])+" "+str(serie["Y"]))
			if(p == serie["Q"]):
				acc += 1
				print("RIGHT!")
			else:
				print("WRONG!")
		return round((acc/acumulator)*100,1)