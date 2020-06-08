# %%
import csv
import math
import random
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def loadCsv(filename):

    data = pd.read_csv("pulsar_stars.csv")
    dataset = data.values.tolist()
    return dataset
   
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)


def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries


def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries


def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1/(math.sqrt(2*math.pi)*stdev))*exponent



def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities    
    
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel



def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions



def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet)))*100.0


def main():
    start = time.time()
    filename = 'pulsar_stars.csv'    
    dataset = loadCsv(filename)
    trainingSet, testSet = train_test_split(dataset, test_size = 0.25)    
            
    
    summaries = summarizeByClass(trainingSet)
    
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)    
    testSet_confusion_matrix = pd.DataFrame(testSet)
    testSet_confusion_matrix = testSet_confusion_matrix.iloc[:,-1]    
    testSet_confusion_matrix = testSet_confusion_matrix.values.tolist()
    cm = confusion_matrix(testSet_confusion_matrix, predictions)    
    
    end = time.time()    
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]

    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    Precision = TP / (TP+FP)
    Recall = TP / (TP+FN)
    

    print('-------------------------------------------------------------')
    print('Naive Bayes ----- >')
    print('-------------------------------------------------------------')
    print(f'Total length of dataset ----- > {len(dataset)}')    
    print(f'Length of Training set ------ > {len(trainingSet)}')    
    print(f'Length of Testing set ------ > {len(testSet)}')        
    print(f' Confusion Matrix for Naive Bayes:' )
    print(cm)
    print('Accuracy of Naive Bayes algorithm: {0}%'.format(accuracy))
    print('\n')    
    print(f'Sensitivity obtained for Naive bayes : {Sensitivity}')
    print(f'Specificity obtained for Naive bayes : {Specificity}\n')
    print(f'Precision : {Precision}')
    print(f'Recall : {Recall} \n')

    print(f'Time taken to model Naive Bayes ------- > {end-start}s \n')
       
main()



# %%
import csv
import math
import random
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def loadCsv(filename):

    data = pd.read_csv("pulsar_stars.csv")
    #data.iloc[:,-1] = data.iloc[:,-1].map({'M':1,'B':0})
    dataset = data.values.tolist()
    return dataset
