import pandas as p
import sys
from ann import ann
#from data_plotter import data_plotter

train = p.read_csv("train.csv",usecols=["X","Y","Q"])
test = p.read_csv("test.csv",usecols=["X","Y","Q"])

machine = ann()
machine.create_ann(
	input_nodes=2,
	hidden_layer_lenght=5,
	number_hidden_layers=5,
	output_nodes=1)

print(train)
machine.train(train[['X','Y']],train[['Q']])
machine.test(test[['X','Y']],test[['Q']])