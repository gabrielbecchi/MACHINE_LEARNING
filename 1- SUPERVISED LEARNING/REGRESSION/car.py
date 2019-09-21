from prepareData import prepareData
from crossValidation import crossValidation
import pandas as p
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

columns = ["symboling","losses","make","fuel-type","aspiration",
	"num-of-doors","body-style","drive-wheels","engine-location",
	"wheel-base","length","width","height","curb-weight","engine-type",
	"num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio",
	"horsepower","peak-rpm","city-mpg","highway-mpg","price"]

#df = p.read_csv("DATA/imports_85.csv",header=None,names=columns,na_values="?")
#df = prepareData.reformCategoricalData(df)
#df.to_csv("DATA/imports_85_reformed.csv")
df = p.read_csv("DATA/imports_85_reformed.csv")

kfold = 5
cv = crossValidation(df,kfold)

machine = linear_model.LinearRegression()
print("Ordinary Least Squares - "+str(round(cv.runTest(machine,"price"),3)))

for alpha in np.arange(0.1,1,0.1):
	machine = linear_model.Ridge(alpha=alpha)
	print("Ridge (alpha: "+str(round(alpha,1))+") - "+\
		str(round(cv.runTest(machine,"price"),3)))

for alpha in np.arange(0.1,1,0.1):
	machine = linear_model.Lasso(alpha=alpha,tol=0.1)
	print("Lasso (alpha: "+str(round(alpha,1))+") - "+\
		str(round(cv.runTest(machine,"price"),3)))

for alpha in np.arange(0.1,1,0.1):
	machine = linear_model.ElasticNet(alpha=alpha,tol=0.1)
	print("ElasticNet (alpha: "+str(round(alpha,1))+") - "+\
		str(round(cv.runTest(machine,"price"),3)))
