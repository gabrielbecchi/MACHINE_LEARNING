import pandas as p
import sys
from ann import ann
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from data_plotter import data_plotter

data = p.read_csv("data_cancer.txt",header=None,index_col=0)
data = data.replace("M",1)
data = data.replace("B",0)

x = data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data = p.DataFrame(x_scaled)

train, test = train_test_split(data,test_size=0.5)

machine = ann()
machine.create_ann(
	input_nodes=30,
	hidden_layer_lenght=30,
	number_hidden_layers=4,
	output_nodes=1)

machine.train(train[list(range(1,31))],train[[0]])
machine.test(test[list(range(1,31))],test[[0]])
