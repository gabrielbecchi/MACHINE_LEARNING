import pandas as p
import sys
from decision_tree import decision_tree
from data_plotter import data_plotter

train = p.read_csv("train.csv")
test = p.read_csv("test.csv")

#data_plotter.plot(train)
#print(train)
#sys.exit(0)

machine = decision_tree()
machine.train(train)

#predict = machine.predict(1,1)
predict = machine.test(train)
print(predict)
predict = machine.test(test)
#predict = machine.crossvalidation()
print(predict)