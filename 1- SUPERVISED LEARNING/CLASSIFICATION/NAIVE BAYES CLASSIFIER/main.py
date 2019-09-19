import pandas as p
from naive_bayes_classifier import naive_bayes_classifier

train = p.read_csv("trainBruno.csv")
test = p.read_csv("testBruno.csv")

#print(train)

machine = naive_bayes_classifier()
machine.train(train)

#machine.predict(1,1)
predict = machine.test(test)
#predict = machine.crossvalidation()
print(predict)