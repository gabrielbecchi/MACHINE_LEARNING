import pandas
import random
import numpy as np
import matplotlib.pyplot as plt


#
NUMBER_OF_LINES_TRAIN = int(500)
NUMBER_OF_LINES_TEST = int(500)
#

def compDistance(point1,point2):
	if(len(point1) != len(point2)):
		raise ValueError("Number of dimentions of all points MUST be equal")
		return
	acc = 0
	for i in range(len(point1)):
		acc += np.square(point1[i]-point2[i])
	return np.sqrt(acc)
	#return np.sqrt(np.square(point1[0]-point2[0])+np.square(point1[1]-point2[1]))

def createDataset(sizeSet,centerPoint,radious,xRange,yRange):
	data = pandas.DataFrame(columns=["X","Y","Q"])
	for index in range(sizeSet):
		serie = pandas.Series(index=["X","Y","Q"])
		serie.X = round(random.uniform(xRange[0],xRange[1]),2)
		serie.Y = round(random.uniform(yRange[0],yRange[1]),2)
		if(compDistance(centerPoint,(serie.X,serie.Y)) <= radious):
			serie.Q = 1
		else:
			serie.Q = 0
		data = data.append(serie,ignore_index=True)
	return data

def plot(df):
	df1 = df.loc[df['Q'] == 1]
	df0 = df.loc[df['Q'] == 0]
	plt.plot(df1.X,df1.Y,'r.')
	plt.plot(df0.X,df0.Y,'b.')
	plt.show()

df_train = createDataset(NUMBER_OF_LINES_TRAIN,(0,0),2.5,[-10,10],[-10,10])
df_test = createDataset(NUMBER_OF_LINES_TEST,(0,0),2.5,[-10,10],[-10,10])
df_train.to_csv("train.csv")
df_test.to_csv("test.csv")
