import matplotlib.pyplot as plt

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