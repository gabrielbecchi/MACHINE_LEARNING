import pandas as p
import sys
from gradient_descend import gradient_descend
from data_plotter import data_plotter

train = p.read_csv("train.csv")
test = p.read_csv("test.csv")

#data_plotter.plot(test)
#print(test)
#sys.exit(0)

machine = gradient_descend()
machine.train(train)

#predict = machine.predict(1,1)
predict = machine.test(train)
print("CROSS-VALIDATION: "+str(predict))
predict = machine.test(test)
#predict = machine.crossvalidation()
print("TEST SET: "+str(predict))

#PLOT CLASSIFICATION
#sys.exit()
print("PLOT CLASSIFIER")
map_train = p.read_csv("map_train.csv")
map_df = data_plotter.classification_map_generator()
map_df = machine.build_map(map_df)
data_plotter.classification_plot(map_train,map_df)