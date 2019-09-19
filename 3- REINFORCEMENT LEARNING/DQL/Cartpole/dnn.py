from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras import optimizers
from keras import backend as K
import numpy as np

#kernel_initializer=rbm
#shape = (input units)
#def rbm(shape, dtype=None):
#	#input(shape)
#	#print("AAA")
#	return K.random_normal(shape, dtype=dtype)

class dnn:
	def create_model(self,input_dim,output_dim,number_layers,units_per_layer):
		self.model = Sequential()
		
		#First layer with input dimention
		#kernel_initializer=rbm
		self.model.add(Dense(units=units_per_layer,activation='tanh',input_dim=input_dim))
		#Loop for adding other hidden layers
		if(number_layers < 2):
			raise ValueError("Number of layers must be at least 2")
		for i in range(0,number_layers-2):
			self.model.add(Dense(units=units_per_layer,activation='tanh'))
		#Final layer
		self.model.add(Dense(units=output_dim,activation='tanh'))
		self.compile_model()

	def compile_model(self,lr=0.5):
		#sgd = optimizers.SGD(lr=0.00025, decay=1e-2, momentum=0.2, nesterov=True)
		#self.model.compile(optimizer=sgd,loss='mse')
		self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=lr))
		#print("END")

	def pretrain_model(self,x,y):
		self.compile_model(0.01)
		self.model.fit(x, y, epochs=5,verbose=1);
		self.compile_model()

	def train_model(self,x,y):
		self.model.fit(x, y, epochs=1,verbose=1)

	def predict(self,x):
		x = np.asarray([x])
		y = self.model.predict(x)
		y[0] = y[0].round(3)
		#print(y[0])
		return y[0]

	def save_model(self):
		model_json = self.model.to_json()
		with open("model.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		self.model.save_weights("model.h5")
		print("Saved model to disk")

	def load_model(self):
		# load json and create model
		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(loaded_model_json)
		# load weights into new model
		self.model.load_weights("model.h5")
		self.compile_model()
		print("Loaded model from disk")
