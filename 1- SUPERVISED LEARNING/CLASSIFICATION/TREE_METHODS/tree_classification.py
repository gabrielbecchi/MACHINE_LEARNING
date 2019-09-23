from prepareData import prepareData
from crossValidation import crossValidation
import pandas as p
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble

kfold = 5
cv = crossValidation(df,kfold)

for depth in range(1,25):
	machine = tree.DecisionTreeRegressor(max_depth=depth)
	print("DecisionTree - Depth "+str(round(depth,1))+" - "+\
		str(round(cv.runTest(machine,labelColumn),3)))

for depth in range(1,25):
	machine = ensemble.RandomForestRegressor(n_estimators=100,max_depth=depth)
	print("RandomForest - Depth "+str(round(depth,1))+" - "+\
		str(round(cv.runTest(machine,labelColumn),3)))

for depth in range(1,25):
	machine = ensemble.ExtraTreesRegressor(n_estimators=100,max_depth=depth)
	print("ExtraTreesRegressor - Depth "+str(round(depth,1))+" - "+\
		str(round(cv.runTest(machine,labelColumn),3)))