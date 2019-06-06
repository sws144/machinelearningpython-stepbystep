# -*- coding: utf-8 -*-
"""
Spyder Editor

Machine learning step by step

https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
"""

#1.2 Start Python and Check Versions 

#check library versions
import sys
print('Python: {}'.format(sys.version))
#scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
#numpy
import numpy
print('numpy: {}' .format(numpy.__version__))
#matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
#pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
#scikit learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


#2. Load The Data
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 2.2 Load Dataset

#load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# 3. Summarize the Dataset
#shape
print(dataset.shape)
#head
print(dataset.head(20))
#3.3 statistical summary
print(dataset.describe())
#class distribution
print(dataset.groupby('class').size())
#box and whisker plots 
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()

## 4.1 univariate plots
#histograms
dataset.hist()

#4.2 multivariate plots
# scatter plot matrix
scatter_matrix(dataset)

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size = validation_size, random_state = seed)
# Test options and evaluation matrix
seed = 7
scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma = 'auto')))
#evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()