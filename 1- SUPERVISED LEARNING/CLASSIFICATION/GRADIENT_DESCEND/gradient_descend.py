import random
import math
from datetime import datetime

class gradient_descend:
	def artificial_neuron_start(self):
		random.seed(datetime.now())
		self.w0 = round(random.uniform(0,1),2)
		self.w1 = round(random.uniform(0,1),2)
		self.w2 = round(random.uniform(0,1),2)

	def update_weights(self):
		self.w0 += round(self.d_w0,2)
		self.w1 += round(self.d_w1,2)
		self.w2 += round(self.d_w2,2)

	def computate_new_weights(self,output,target,x,y):
		#print("*")
		#print(target)
		#input(output)
		self.d_w0 = round(self.learning_rate*(target-output)*1,2)
		self.d_w1 = round(self.learning_rate*(target-output)*x,2)
		self.d_w2 = round(self.learning_rate*(target-output)*y,2)

	def computate_output(self,x,y):
		w = self.w0*1
		w += self.w1*x
		w += self.w2*y
		output = round(math.exp(w)/(math.exp(w)+1),3)
		#input(output)
		return (output+1)

	def train(self,traindf,learning_rate=0.1,numb_iterations=100):
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
		print(self.w0)
		print(self.w1)
		print(self.w2)


	def test(self,testdf):
		acc = 0
		for index, rowSerie in testdf.iterrows():
			if(int(round(self.computate_output(rowSerie["X"],rowSerie["Y"]))) == rowSerie["Q"]):
				acc += 1
		return round(((acc/len(testdf))*100),2)

	def build_map(self,map_df):
		for index, rowSerie in map_df.iterrows():
			rowSerie["Q"] = int(round(self.computate_output(rowSerie["X"],rowSerie["Y"])))
			map_df.ix[index] = rowSerie
			#input(map_df)
		return map_df


