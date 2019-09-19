import pandas
import random
import numpy as np
import matplotlib.pyplot as plt

#
NUMBER_OF_LINES_TRAIN = int(500)
NUMBER_OF_LINES_TEST = int(500)
#

#Quadrante
'''
def classifyPoint(x,y):
	if(x > 0):
		if(y > 0):
			return 1
		else:
			return 4
	else:
		if(y > 0):
			return 2
		else:
			return 3
'''
#'''
#Function
def classifyPoint(x,y):
	if(y < (0.05*(x)**2+0.1*(x)**3)):
		return 1
	else:
		return 0
#'''
'''
def classifyPoint(x,y):
	if(y < x):
		return 2
	else:
		return 1
'''

#Gerar Ponto 2D
def generate_point():
	x = 0
	y = 0
	while(x == 0):
		x = round(random.uniform(-10,10),2)
	while(y == 0):
		y = round(random.uniform(-10,10),2)
	return x,y

def plot_distribution(df):	
	fig = plt.figure()
	#plt.axis([-10,10,-10,10])
	ax = fig.add_subplot(1, 1, 1)
	# Move left y-axis and bottim x-axis to centre, passing through (0,0)
	ax.spines['left'].set_position('center')
	ax.spines['bottom'].set_position('center')

	# Eliminate upper and right axes
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')

	# Show ticks in the left and lower axes only
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

	dft = df[(df.Q == 0)]
	plt.plot(dft["X"],dft["Y"], 'yo')
	dft = df[(df.Q == 1)]
	plt.plot(dft["X"],dft["Y"], 'ro')
	dft = df[(df.Q == 2)]
	plt.plot(dft["X"],dft["Y"], 'bo')
	dft = df[(df.Q == 3)]
	plt.plot(dft["X"],dft["Y"], 'go')
	#dft = df[(df.Q == 4)]
	#plt.plot(dft["X"],dft["Y"], 'yo')

	plt.show()

def creat_train():
	df = pandas.DataFrame(columns=["X","Y","Q"])
	for line in range(NUMBER_OF_LINES_TRAIN):
		x,y = generate_point()
		quadrante = classifyPoint(x,y)

		line = pandas.Series(name = line)
		line["X"] = x
		line["Y"] = y
		line["Q"] = int(quadrante)
		line["E"] = 1-int(quadrante)
		df = df.append(line)
		#print(line)

	plot_distribution(df)
	df.to_csv("train.csv",index=False)

def creat_test():
	df = pandas.DataFrame(columns=["X","Y","Q"])
	for line in range(NUMBER_OF_LINES_TEST):
		x,y = generate_point()
		quadrante = classifyPoint(x,y)

		line = pandas.Series(name = line)
		line["X"] = x
		line["Y"] = y
		line["Q"] = int(quadrante)
		line["E"] = 1-int(quadrante)
		df = df.append(line)
		#print(line)

	plot_distribution(df)
	df.to_csv("test.csv",index=False)

def creat_prot_map():
	df = pandas.DataFrame(columns=["X","Y","Q"])
	it = 0
	for x in np.arange(-10.5,10.5,1):
		for y in np.arange(-10.5,10.5,1):
			line = pandas.Series(name = it)
			it += 1
			line["X"] = x
			line["Y"] = y
			line["Q"] = classifyPoint(x,y)
			df = df.append(line)
			#print(str(x)+str(y))
	#print(it)
	df.to_csv("map_train.csv",index=False)


creat_train()
creat_test()
creat_prot_map()