import matplotlib.pyplot as plt
import pandas as p
import numpy as np
import os

class data_plotter:
	def plot(df):	
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

		dft = df[(df.Q == 1)]
		plt.plot(dft["X"],dft["Y"], 'ro')
		dft = df[(df.Q == 2)]
		plt.plot(dft["X"],dft["Y"], 'bo')
		dft = df[(df.Q == 3)]
		plt.plot(dft["X"],dft["Y"], 'go')
		dft = df[(df.Q == 4)]
		plt.plot(dft["X"],dft["Y"], 'yo')

		plt.show()

	def classification_plot(function_df,map_df):
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

		df = function_df
		dft = df[(df.Q == 1)]
		plt.plot(dft["X"],dft["Y"], 'ro')
		dft = df[(df.Q == 2)]
		plt.plot(dft["X"],dft["Y"], 'co')

		df = map_df
		dft = df[(df.Q == 1)]
		plt.plot(dft["X"],dft["Y"], 'mv')
		dft = df[(df.Q == 2)]
		plt.plot(dft["X"],dft["Y"], 'bv')
		plt.show()

	def classification_map_generator():
		if(os.path.isfile("map.csv")):
			map_df = p.read_csv("map.csv")
			return map_df

		df = p.DataFrame(columns=["X","Y","Q"])
		it = 0
		for x in np.arange(-10,10,1):
			for y in np.arange(-10,10,1):
				line = p.Series(name = it)
				it += 1
				line["X"] = x
				line["Y"] = y
				df = df.append(line)
				#print(str(x)+str(y))
		#print(it)
		df.to_csv("map.csv",index=False)
		return df
