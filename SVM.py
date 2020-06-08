# %%
import sys
from sklearn.model_selection import RepeatedKFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import operator
#from INSE6110.FinalCode.PreprocessForSpamAssasinDatabase import StartPreprocessing
import numpy as np
sys.path.append("../ Untitled Folder 1")
from SimulateResults import * 
#from KNNImplementation import *
#from NaiveBayesImplementation import *
#    from DecisionTreeImplementation import  *
#from SVMImplementation import * 

#define global variables
folds = 10
repeats = 1
K = 5
simulateResults = SimulateResults()
label = [0, 1]

# #step-1 generate preprocess data

#step-2 generate data for spambase data and apply KNN
#print("Step-2: generating dataset for spambase")
dataset = pd.read_csv("pulsar_stars.csv",names=col_Names)
X = np.array(dataset.iloc[:, 0:48])
Y = np.array(dataset.iloc[:, -1])

#genrate k folds
kf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=None)

#get predictions for each fold
print("processing k folds for spambase")
testRecords = []
#predictedRecordsForKNN = []
#predictedRecordsForNB = []
#predictedRecordsForDT = []
predictedRecordsForSVM = []

foldIndex = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # GET SVM Prediction
    print("         Getting SVM results...")
    svm = SupportVectorMachine(X_train, Y_train, X_test)
    predictedRecordsForSVMForTheFold = svm.getPrediction()

    #appends records for displaying results
    for t in Y_test:
        testRecords.append(t)
        
    for t in predictedRecordsForSVMForTheFold:
        predictedRecordsForSVM.append(t)


#get confusionmatrix

cmForSVM = simulateResults.getConfusionMatrix(testRecords, predictedRecordsForSVM)
print("Result for Spambase - SVM")
print(cmForSVM)
print(simulateResults.getAccuracy(testRecords,predictedRecordsForSVM))

#plot confusionmatrix
#simulateResults.plot_confusion_matrix(cmForKNN,label,'Confusion matrix for KNN - Spambase')
#simulateResults.plot_confusion_matrix(cmForNB,label,'Confusion matrix for NB - Spambase')
#simulateResults.plot_confusion_matrix(cmForKNN,label,'Confusion matrix for Decision Tree - Spambase')
#simulateResults.plot_confusion_matrix(cmForNB,label,'Confusion matrix for SVM - Spambase')

#Step 3: Preprocess Data
print("Preprocess Data")
processedData = []

#sdr,ldr,tdr= simulateResults.calculate(cmForKNN)
#processedData.append({"classifier":"KNN", "sdr":sdr, "ldr":ldr, "tdr":tdr})

#sdr,ldr,tdr= simulateResults.calculate(cmForNB)
#processedData.append({"classifier":"Naive Bayes", "sdr":sdr, "ldr":ldr, "tdr":tdr})

#sdr,ldr,tdr= simulateResults.calculate(cmForDT)
#processedData.append({"classifier":"Decision Tree", "sdr":sdr, "ldr":ldr, "tdr":tdr})

sdr,ldr,tdr= simulateResults.calculate(cmForSVM)
processedData.append({"classifier":"SVM", "sdr":sdr, "ldr":ldr, "tdr":tdr})

print(processedData)

#Step 4: Show graph
#simulateResults.generateGraph(processedData)


# %%
sys.path

# %%
sys.executable

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
