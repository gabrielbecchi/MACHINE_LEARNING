import pandas as p
import numpy as np
from sklearn import svm
from crossValidation import crossValidation

import warnings
warnings.filterwarnings("ignore")

#p.set_option('display.max_rows', 500)
#p.set_option('display.max_columns', 500)
#p.set_option('display.width', 1000)

name = ['index','label'] + [str(i) for i in list(np.arange(0,30))]
df = p.read_csv("DATA/cancer.csv",names=name, index_col='index')
df_label = df['label']
df = df.drop('label',1)
df = (df-df.min())/(df.max()-df.min())
df['label'] = df_label

df.loc[df['label'] == 'B', 'label'] = 0
df.loc[df['label'] == 'M', 'label'] = 1
labelColumn = 'label'

cv = crossValidation(df,5)
machine = svm.LinearSVC(max_iter=20000,random_state=0,tol=0.01)
print("SVM - Linear ")
print(cv.runTestClassification(machine,labelColumn))

cv = crossValidation(df,5)
machine = svm.SVC(kernel='rbf',max_iter=20000,random_state=0,tol=0.01)
print("SVM - RBF ")
print(cv.runTestClassification(machine,labelColumn))

cv = crossValidation(df,5)
machine = svm.NuSVC(nu=0.1,kernel='rbf',max_iter=20000,random_state=0,tol=0.01)
print("SVM - Nu (RBF)")
print(cv.runTestClassification(machine,labelColumn))
