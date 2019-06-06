# -*- coding: utf-8 -*-
"""
Created on Sat May  4 10:29:38 2019

@author: SW

https://machinelearningmastery.com/load-machine-learning-data-python/
"""

#Load CSV (using python)
import csv
import numpy
filename = 'pima-indians-diabetes.data.csv'
raw_data = open('C:\Stuff\Important\CareerNCollege\Ad Hoc\\' + filename, 'rt')
reader = csv.reader(raw_data, delimiter = ',', quoting=csv.QUOTE_NONE)
x=list(reader)
data = numpy.array(x).astype('float')
print(data.shape)

#Load CSV
import numpy
filename = 'pima-indians-diabetes.data.csv'
raw_data = open('C:\Stuff\Important\CareerNCollege\Ad Hoc\\' + filename, 'rt')
data = numpy.loadtxt(raw_data, delimiter=",")
print(data.shape)

#Load CSV from URL using NumPy
from numpy import loadtxt
from urllib.request import urlopen
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
raw_data = urlopen(url)
dataset = loadtxt(raw_data, delimiter = ",")
print(dataset.shape)

#Load CSV using Pandas
import pandas
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
filename = 'pima-indians-diabetes.data.csv'
data = pandas.read_csv('C:\Stuff\Important\CareerNCollege\Ad Hoc\\' + filename, names=names)
print(data.shape)
