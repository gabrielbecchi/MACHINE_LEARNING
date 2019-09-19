from prepareData import prepareData
from crossValidation import crossValidation
import pandas as p
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

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

for i in range(kfold):
	train, test = cv.getNextSet()
	input(train)

