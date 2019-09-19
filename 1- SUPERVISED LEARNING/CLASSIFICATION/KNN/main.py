import pandas as p
from knn import knn

train = p.read_csv("testBruno.csv")
test = p.read_csv("trainBruno.csv")

#print(train)

machine = knn()
machine.train(train,3,weightVoting=False)
predict = machine.test(test)
#predict = machine.crossvalidation()
print(predict)