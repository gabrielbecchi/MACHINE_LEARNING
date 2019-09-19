import pandas as p
import sys
from brf import brf
#from data_plotter import data_plotter

train = p.read_csv("train.csv")
test = p.read_csv("test.csv")

machine = ann()
machine.create_ann(
	input_nodes=1,
	hidden_layer_lenght=4,
	number_hidden_layers=4,
	output_nodes=2)

#machine.computate_output([1,1])
machine.train(train[['X','Y']],train[['Q','E']])