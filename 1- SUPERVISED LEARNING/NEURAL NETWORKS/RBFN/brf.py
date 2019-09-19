import random
import math
import time
import numpy

random.seed(time.time())
#-------------------
class ann_funcs:
	def sigmoid(h):
		return round(1/(1+math.exp(-1*h)),3)

	def deriv_sigmoid(h):
		return round(ann_funcs.sigmoid(h)*(1-ann_funcs.sigmoid(h)),3)

	def weight_sum(input_list,weight_list):
		acc = 0
		for i in range(len(input_list)):
			acc += input_list[i]*weight_list[i]
		return acc

#-------------------
class input_node():
	def run(self,input_value):
		return input_value

#-------------------
class generic_node:
	def __init__(self,hidden_layer_lenght):
		self.list_wights = []
		#w0 == bias
		for weight_numb in range(hidden_layer_lenght+1):
			self.list_wights.append(round(random.uniform(-1,1),3))

	def run(self,input_value):
		if(type(input_value) != list):
			list_input = [1,input_value]
		else:
			list_input = [1]+input_value
		self.list_input = list_input
		self.output = ann_funcs.sigmoid(ann_funcs.weight_sum(list_input,self.list_wights))
		return self.output

	def update_weights(self,learning_rate):
		for i in range(len(self.list_wights)):
			self.list_wights[i] -= learning_rate*(self.error*self.list_input[i])

	def generate_backprop_error(self):
		error_per_node = []
		for i in range(1,len(self.list_wights)):
			error_per_node.append(round(self.error*self.list_wights[i],3))
		return error_per_node

#-------------------
class hidden_node(generic_node):
	def computate_error(self,propagated_error):
		self.error = self.output*(1-self.output)*propagated_error
		return self.error

#-------------------
class output_node(generic_node):
	def computate_error(self,expected):
		self.error = self.output*(1-self.output)*(self.output-expected)
		return self.error

#-------------------
class brf:
	#-------------------
	def create(self,input_nodes,hidden_layer_lenght,number_hidden_layers,output_nodes):
		self.list_layers = []

		#Input Layer
		new_hidden_layer = []
		for i in range(input_nodes):
			new_hidden_layer.append(input_node())
		self.list_layers.append(new_hidden_layer)

		#Hidden Layers
		#First Hidden Layer
		new_hidden_layer = []
		for neuron in range(hidden_layer_lenght):
			new_hidden_layer.append(hidden_node(input_nodes))
		self.list_layers.append(new_hidden_layer)
		#Other Hidden Layers
		for hl_numb in range(number_hidden_layers-1):
			new_hidden_layer = []
			for neuron in range(hidden_layer_lenght):
				new_hidden_layer.append(hidden_node(hidden_layer_lenght))
			self.list_layers.append(new_hidden_layer)
		
		#Output Layer
		new_hidden_layer = []
		for i in range(output_nodes):
			new_hidden_layer.append(output_node(hidden_layer_lenght))
		self.list_layers.append(new_hidden_layer)		

	#-------------------
	def computate_output(self,list_parameters):
		#INPUT LAYER
		new_parameters = []
		for node in range(len(self.list_layers[0])):
			new_parameters.append(self.list_layers[0][node].run(list_parameters[node]))
		list_parameters = new_parameters

		#OTHER LAYERS
		for layer in range(1,len(self.list_layers)):
			new_parameters = []
			for node in range(len(self.list_layers[layer])):
				new_parameters.append(self.list_layers[layer][node].run(list_parameters))
			list_parameters = new_parameters
		#print(list_parameters)
		return list_parameters

	#-------------------
	def run_iteration(self,parameters,targets,learning_rate):
		#Forward Feeding
		output = self.computate_output(parameters)

		#Backpropagation
		#Output Layer
		error_in_layer = []
		for i in range(len(self.list_layers[-1])):
			self.list_layers[-1][i].computate_error(targets[i])
			error_in_layer.append(self.list_layers[-1][i].generate_backprop_error())
		error_in_layer = numpy.matrix(error_in_layer).sum(axis=0).tolist()[0]

		#Hidden Layer
		for layer in range(len(self.list_layers)-2,0,-1):
			new_error_in_layer = []
			for i in range(len(self.list_layers[layer])):
				self.list_layers[layer][i].computate_error(error_in_layer[i])
				new_error_in_layer.append(self.list_layers[layer][i].generate_backprop_error())
			error_in_layer = numpy.matrix(new_error_in_layer).sum(axis=0).tolist()[0]

		#UPDATE WEIGHTS
		for layer in range(1,len(self.list_layers)):
			for node in self.list_layers[layer]:
				node.update_weights(learning_rate)

		#if(self.i != 10):
		#	return False
		#STATS
		#print("P:"+str(parameters))
		#print("O:"+str(output))
		#print("T:"+str(targets))
		#print("----")
		if(round(output[0],0)==targets[0]):
			return True
		return False

	def classify(self,parameters,targets):
		output = self.computate_output(parameters)
		if(round(output[0],0)==targets[0]):
			return True
		return False

	def train(self,train_df,train_labels,learning_rate=0.5,numb_epoch=5):
		self.i = 0
		for iteration in range(numb_epoch):
			acc = 0
			for index in range(train_df.shape[0]):
				parameters = list(train_df.iloc[index].values)
				targets = list(train_labels.iloc[index].values)
				if(self.run_iteration(parameters,targets,learning_rate)):
					acc +=1
			#if(self.i == 10):
			#	self.i = 0
			#	print(acc/train_df.shape[0])
			#	print("----------------")
			#else:
			#	self.i += 1
			print("---------")
			print(acc/train_df.shape[0])
			print("---------")

	def test(self,train_df,train_labels):
		acc = 0
		for index in range(train_df.shape[0]):
			parameters = list(train_df.iloc[index].values)
			targets = list(train_labels.iloc[index].values)
			if(self.classify(parameters,targets)):
				acc +=1
		print("-----TEST-----")
		print(acc/train_df.shape[0])
		print("--------------")


