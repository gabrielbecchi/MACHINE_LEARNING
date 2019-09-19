import pandas
import operator

class knn:
	def train(self,trainDF,k,weightVoting=True):
		self.trainDF = trainDF
		self.k = k
		self.weightVoting = weightVoting

	#Aux Functions
	def distance_Function(x1,y1,x2,y2):
		return (((x1-x2)**2)+((y1-y2))**2)**0.5

	#Simple Voting
	def simple_voting(listPoints):
		dicVotes = {}
		for serie in listPoints:
			if(serie["Q"] in dicVotes.keys()):
				dicVotes[serie["Q"]] += 1
			else:
				dicVotes[serie["Q"]] = 1
		#Get Most Votes
		sorted_votes = sorted(dicVotes.items(), key=operator.itemgetter(1))
		acc = round(sorted_votes[-1][1]/len(listPoints),3)

		return (sorted_votes[-1][0],acc*100)

	def normalizer(value,shortest,longest):
		discount = (value-shortest) / (longest-shortest)
		return round(1 - discount,3)

	def weight_voting(listPoints,listSortDistance):
		dicVotes = {}
		accumulator = 0
		for x in range(len(listPoints)):
			serie = listPoints[x]
			accumulator += knn.normalizer(listSortDistance[x],listSortDistance[-1],listSortDistance[0])
			if(serie["Q"] in dicVotes.keys()):
				dicVotes[serie["Q"]] += knn.normalizer(listSortDistance[x],listSortDistance[-1],listSortDistance[0])
			else:
				dicVotes[serie["Q"]] = knn.normalizer(listSortDistance[x],listSortDistance[-1],listSortDistance[0])
		
		#Get Most Votes
		sorted_votes = sorted(dicVotes.items(), key=operator.itemgetter(1))
		acc = round(sorted_votes[-1][1]/accumulator,3)
		#print(sorted_votes)
		return (sorted_votes[-1][0],acc*100)

	def sortListsBubble(listSortDistance,listPoints):
		if(len(listSortDistance) <= 1):
			return listSortDistance,listPoints
		acertos = 0
		while(acertos != len(listSortDistance)-1):
			acertos = 0
			for i in range(len(listSortDistance)-1):
				if(listSortDistance[i] >= listSortDistance[i+1]):
					acertos += 1
					continue
				else:
					aux1 = listSortDistance[i]
					aux2 = listPoints[i]
					listSortDistance[i] = listSortDistance[i+1]
					listPoints[i] = listPoints[i+1]
					listSortDistance[i+1] = aux1
					listPoints[i+1] = aux2
		return listSortDistance,listPoints

	def predict(self,x,y):
		#Define Near Points
		listSortDistance = []
		listPoints = []
		for index, rowSerie in self.trainDF.iterrows():
			#Iniciando Vetores
			if(len(listPoints) < self.k):
				listPoints.append(rowSerie)
				listSortDistance.append(knn.distance_Function(x,y,rowSerie["X"],rowSerie["Y"]))
				listSortDistance,listPoints = knn.sortListsBubble(listSortDistance,listPoints)
				continue
			newDist = knn.distance_Function(x,y,rowSerie["X"],rowSerie["Y"])
			for i in range(len(listSortDistance)):
				if(newDist < listSortDistance[i]):
					listSortDistance[i] = newDist
					listPoints[i] = rowSerie
					listSortDistance,listPoints = knn.sortListsBubble(listSortDistance,listPoints)
					break
			continue
		if(self.weightVoting):
			return knn.weight_voting(listPoints,listSortDistance)
		else:
			return knn.simple_voting(listPoints)

	def test(self,df):
		acc = 0
		counter = 0
		for index, serie in df.iterrows():
			print(index+1)
			print(str(serie["X"])+" "+str(serie["Y"]))
			counter += 1
			prediction = self.predict(serie["X"],serie["Y"])
			if(prediction[0] == serie["Q"]):
				print("RIGHT!")
				acc += 1
			else:
				print("WRONG!")
		acc = round((acc/counter)*100,2)
		return acc



	def crossvalidation(self):
		return self.test(self.trainDF)




