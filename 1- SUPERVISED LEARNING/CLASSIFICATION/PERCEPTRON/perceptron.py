import random
from datetime import datetime

class perceptron:
	def artificial_neuron_start(self):
		random.seed(datetime.now())
		self.w0 = round(random.uniform(0,1),2)
		self.w1 = round(random.uniform(0,1),2)
		self.w2 = round(random.uniform(0,1),2)

	def update_weights(self):
		self.w0 += self.d_w0
		self.w1 += self.d_w1
		self.w2 += self.d_w2

	def computate_new_weights(self,output,target,x,y):
		self.d_w0 = round(self.learning_rate*(target-output)*1,2)
		self.d_w1 = round(self.learning_rate*(target-output)*x,2)
		self.d_w2 = round(self.learning_rate*(target-output)*y,2)

	def computate_output(self,x,y):
		output = self.w0*1
		output += self.w1*x
		output += self.w2*y
		if(output>=0):
			return 2
		return 1

	def train(self,traindf,learning_rate=0.25,numb_iterations=10):
		self.learning_rate = learning_rate
		self.artificial_neuron_start()
		for inter in range(numb_iterations):
			for index, rowSerie in traindf.iterrows():
				output = self.computate_output(rowSerie["X"],rowSerie["Y"])
				self.computate_new_weights(output,rowSerie["Q"],rowSerie["X"],rowSerie["Y"])
				self.update_weights()
			if(self.test(traindf)==100):
				print("BREAK TRAIN: "+str(inter+1))
				break
		#print(self.w0)
		#print(self.w1)
		#input(self.w2)


	def test(self,testdf):
		acc = 0
		for index, rowSerie in testdf.iterrows():
			if(self.computate_output(rowSerie["X"],rowSerie["Y"])==rowSerie["Q"]):
				acc += 1
		return round(((acc/len(testdf))*100),2)
