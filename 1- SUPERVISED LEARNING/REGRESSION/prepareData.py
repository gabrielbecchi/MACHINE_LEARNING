import pandas as p
import numpy as np

#p.set_option('display.max_rows', 500)
p.set_option('display.max_columns', 500)
#p.set_option('display.width', 1000)

class prepareData:
	def reformCategoricalData(data):
		data = data.dropna().reset_index()
		newDf = p.DataFrame()
		for column,dataType in data.dtypes.items():
			#Numerical
			if(np.issubdtype(dataType,np.int64) or np.issubdtype(dataType,np.float64)):
				newDf[column] = data[column]
				del data[column]
				continue
			#Categorical
			for uniqueValue in data[column].unique():
				newColumn = p.Series()
				for index, value in data[column].iteritems():
					if(value == uniqueValue):
						newColumn.at[index] = 1
					else:
						newColumn.at[index] = 0
				newDf[uniqueValue] = newColumn
		return newDf

