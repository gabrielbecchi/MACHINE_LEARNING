import math
from scipy import optimize
import random

class treeOp:
	def entropy(df):
		entp = 0
		for label in df["Q"].unique():
			prob = len(df.loc[df["Q"] == label])/len(df)
			entp += (-1*(prob*math.log2(prob)))
		#print(entp)
		return entp

	def splitData(listParameters,df):
		df1 = df.loc[df["X"] >= listParameters[0]]
		df1 = df1.loc[df["Y"] >= listParameters[1]]
		df2 = df.drop(df1.index)
		return df1,df2

	def average_entrop(df1,df2):
		size = len(df1)+len(df2)
		entp1 = treeOp.entropy(df1)
		entp2 = treeOp.entropy(df2)
		entp = (entp1*(len(df1)/size))+(entp2*(len(df2)/size))
		return entp


	def minimizeFunc(listParameters,df):
		df1,df2 = treeOp.splitData(listParameters,df)
		entp = treeOp.average_entrop(df1,df2)
		#print(listParameters)
		#print("WEIGHTED AVERAGE "+str(entp))
		return entp

	def size(parameters,df):
		df1,df2 = treeOp.splitData(parameters,df)
		if(len(df1)<len(df2)):
			return len(df1)
		return len(df2)

	def findOptimalParameters(df):
		#Getting First Guess
		minEntropy = 1
		#guess = []
		for i in range(5):
			testGuess = [round(random.uniform(-10,10),2),round(random.uniform(-10,10),2)]
			testGuessEnt = treeOp.minimizeFunc(testGuess,df)
			if(testGuessEnt < minEntropy):
				minEntropy = testGuessEnt
				guess = testGuess
		#print("FINAL GUESS: "+str(guess))
		#First Optimization
		guess = [round(random.uniform(-10,10),2),round(random.uniform(-10,10),2)]
		resp = optimize.minimize(treeOp.minimizeFunc,guess, method='SLSQP', args=(df),
			options={'disp': False,'eps':1,'ftol':0.01,'maxiter':10000},bounds=((-10,10),(-10,10)))
		#Second Optimization
		#print("OPTIMAL 1 "+str(resp.x))
		guess = resp.x
		resp = optimize.minimize(treeOp.minimizeFunc,guess, method='SLSQP', args=(df),
			options={'disp': False,'eps':0.01,'ftol':0.001,'maxiter':10000},bounds=((-10,10),(-10,10)))
		#print("Division!")
		#input("OPTIMAL 2 "+str(resp.x))
		return resp.x

	def defineLabel(df):
		values = df["Q"].value_counts()
		keys = df["Q"].unique()
		mainKey = keys[0]
		for label in keys:
			if(values[label] > values[mainKey]):
				mainKey = label
		return mainKey


class decision_tree:
	def train(self,trainDF,entropyObj=0.2,minimal_porc_lenght=0.005,size_initinal=-1):
		if(size_initinal < 0):
			size_initinal=len(trainDF)
		self.returns = []
		#Checking if Leave
		if(treeOp.entropy(trainDF) <= entropyObj or len(trainDF) <= minimal_porc_lenght*size_initinal):
			#input(treeOp.entropy(trainDF))
			self.nodeType = "leave"
			self.returns.append(treeOp.defineLabel(trainDF))
			print("---LEAVE---")
			print("LABEL: "+str(treeOp.defineLabel(trainDF)))
			print("NUMB TRAIN SAMPLES: "+str(len(trainDF)))
			print("ENTROPY: "+str(treeOp.entropy(trainDF)))
			input()
			return
		#Computating Node
		self.divParameters = treeOp.findOptimalParameters(trainDF)
		df1,df2 = treeOp.splitData(self.divParameters,trainDF)
		#Caso Nao haja Ganho de Entropia
		if(len(df1)==0):
			self.nodeType = "leave"
			self.returns.append(treeOp.defineLabel(df2))
			print("---LEAVE---")
			print("LABEL: "+str(treeOp.defineLabel(df2)))
			print("NUMB TRAIN SAMPLES: "+str(len(df2)))
			print("ENTROPY: "+str(treeOp.entropy(df2)))
			input()
			return
		if(len(df2)==0):
			self.nodeType = "leave"
			self.returns.append(treeOp.defineLabel(df1))
			print("---LEAVE---")
			print("LABEL: "+str(treeOp.defineLabel(df1)))
			print("NUMB TRAIN SAMPLES: "+str(len(df1)))
			print("ENTROPY: "+str(treeOp.entropy(df1)))
			input()
			return
		print("---DIVISION---")
		print("SIDE 1: "+str(len(df1)))
		print("SIDE 2: "+str(len(df2)))
		print("AVERAGE ENTROPY: "+str(treeOp.average_entrop(df1,df2)))
		print("DIV PARAMETERS: "+str(self.divParameters))
		input()
		self.nodeType = "split"
		#Analizando DF1
		self.returns.append(decision_tree())
		self.returns[0].train(df1,entropyObj=entropyObj,size_initinal=size_initinal)
		#Analizando DF2
		self.returns.append(decision_tree())
		self.returns[1].train(df2,entropyObj=entropyObj,size_initinal=size_initinal)
		return

	def predict(self,x,y):
		if(self.nodeType == "leave"):
			return self.returns[0]
		if(x >= self.divParameters[0] and y >= self.divParameters[1]):
			return self.returns[0].predict(x,y)
		else:
			return self.returns[1].predict(x,y)

	def test(self,df):
		acc = 0
		counter = 0
		for index, serie in df.iterrows():
			#print(index+1)
			#print(str(serie["X"])+" "+str(serie["Y"]))
			counter += 1
			prediction = self.predict(serie["X"],serie["Y"])
			if(prediction == serie["Q"]):
				#print("RIGHT!")
				acc += 1
			#else:
				#print("WRONG!")
		acc = round((acc/counter)*100,2)
		return acc

