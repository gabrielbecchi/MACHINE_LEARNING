import pandas as p
import sys
from ann import ann
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from data_plotter import data_plotter

def poker_transformation(value):
	if(value == 0):
		return 0,0,0,0
	if(value == 1):
		return 0,0,0,1
	if(value == 2):
		return 0,0,1,0
	if(value == 3):
		return 0,0,1,1
	if(value == 4):
		return 0,1,0,0
	if(value == 5):
		return 0,1,0,1
	if(value == 6):
		return 0,1,1,0
	if(value == 7):
		return 0,1,1,1
	if(value == 8):
		return 1,0,0,0
	if(value == 9):
		return 1,0,0,1

def prepare_df(old_df):
	df = p.DataFrame()
	for index, row in old_df.iterrows():
		r = poker_transformation(row[10])
		row[10] = r[0]
		row[11] = r[1]
		row[12] = r[2]
		row[13] = r[3]
		df = df.append(row)
	return df


#PREPARE TRAIN
#train = p.read_csv("poker_train.csv",header=None,index_col=None)
#train = prepare_df(train)
#train.to_csv("poker_train_corrected.csv",index=False)
#sys.exit()

#PREPARE TEST
#test = p.read_csv("poker_test.csv",header=None,index_col=None)
#test = prepare_df(test)
#test.to_csv("poker_test_corrected.csv",index=False)
#sys.exit()

train = p.read_csv("poker_train_corrected.csv",header=None,index_col=None)
test = p.read_csv("poker_test_corrected.csv",header=None,index_col=None)

input(test)
sys.exit()


machine = ann()
machine.create_ann(
	input_nodes=10,
	hidden_layer_lenght=1,
	number_hidden_layers=5,
	output_nodes=4)

machine.train(train[list(range(0,10))],train[list(range(10,14))])
machine.test(test[list(range(0,10))],test[list(range(10,14))])