import pandas as p
import numpy as np
from sklearn import svm, neighbors
from crossValidation import crossValidation

import warnings
warnings.filterwarnings("ignore")

#p.set_option('display.max_rows', 500)
#p.set_option('display.max_columns', 500)
#p.set_option('display.width', 1000)

df = p.read_csv("DATA/iris.csv",names=[0,1,2,3,'label'])
labelColumn = 'label'

cv = crossValidation(df,5)
machine = neighbors.KNeighborsClassifier(n_neighbors=3)
print("KNN - K=3 ")
print(cv.runTestClassification(machine,labelColumn))

cv = crossValidation(df,5)
machine = neighbors.KNeighborsClassifier(n_neighbors=5)
print("KNN - K=5 ")
print(cv.runTestClassification(machine,labelColumn))

cv = crossValidation(df,5)
machine = neighbors.KNeighborsClassifier(n_neighbors=7)
print("KNN - K=7")
print(cv.runTestClassification(machine,labelColumn))

cv = crossValidation(df,5)
machine = svm.LinearSVC(max_iter=5000,random_state=0,tol=0.01)
print("SVM - Linear ")
print(cv.runTestClassification(machine,labelColumn))

cv = crossValidation(df,5)
machine = svm.SVC(kernel='rbf',max_iter=5000,random_state=0,tol=0.01)
print("SVM - RBF ")
print(cv.runTestClassification(machine,labelColumn))

cv = crossValidation(df,5)
machine = svm.NuSVC(nu=0.1,kernel='rbf',max_iter=5000,random_state=0,tol=0.01)
print("SVM - Nu (RBF)")
print(cv.runTestClassification(machine,labelColumn))
