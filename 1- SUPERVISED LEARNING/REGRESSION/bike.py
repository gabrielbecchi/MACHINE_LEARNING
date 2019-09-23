from prepareData import prepareData
from crossValidation import crossValidation
import pandas as p
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

df = p.read_csv("DATA/bike_hour.csv",index_col=0)
df = df.drop(columns=['dteday','casual','registered'])
#df = prepareData.reformCategoricalData(df)

kfold = 5
cv = crossValidation(df,kfold)

machine = linear_model.LinearRegression()
print("Ordinary Least Squares - "+str(round(cv.runTest(machine,"cnt"),3)))

for alpha in np.arange(0.5,5,0.5):
	machine = linear_model.Ridge(alpha=alpha)
	print("Ridge (alpha: "+str(round(alpha,1))+") - "+\
		str(round(cv.runTest(machine,"cnt"),3)))

for alpha in np.arange(0.1,1.1,0.1):
	machine = linear_model.Lasso(alpha=alpha,tol=0.1)
	print("Lasso (alpha: "+str(round(alpha,1))+") - "+\
		str(round(cv.runTest(machine,"cnt"),3)))

for alpha in np.arange(0.1,1.1,0.1):
	machine = linear_model.ElasticNet(alpha=alpha,tol=0.1)
	print("ElasticNet (alpha: "+str(round(alpha,1))+") - "+\
		str(round(cv.runTest(machine,"cnt"),3)))

