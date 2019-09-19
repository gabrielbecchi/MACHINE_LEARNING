import pandas as p
import sys
from perceptron import perceptron
from data_plotter import data_plotter

train = p.read_csv("train.csv")
test = p.read_csv("test.csv")

data_plotter.plot(train)
print(train)
sys.exit(0)

machine = perceptron()
machine.train(train)

#predict = machine.predict(1,1)
predict = machine.test(train)
print("CROSS-VALIDATION: "+str(predict))
predict = machine.test(test)
#predict = machine.crossvalidation()
print("TEST SET: "+str(predict))