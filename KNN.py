# %%
import csv
import random
import time
import pandas as pd
from sklearn.metrics import confusion_matrix

def handleDataset(filename, split, trainingSet, testSet):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(8):
                dataset[x][y] = float(dataset[x][y])  
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
                


import math
def euclideanDistance(instance1, instance2, length):    
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


import operator 
def getKNeighbors(trainingSet, testInstance, k):
    distances = []
    
    length = len(testInstance)-1    # Total rows of testSet
    for x in range(len(trainingSet)): # For each row in trainingSet, calculate the distance from each of the training set.
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1)) # Sort by distance ------ > i.e lowest distances will be placed first
    neighbors = []
    for x in range(k): # --------- > picks 3 smallest distances ------ > 3 nearest Neighbours ----- > First 3 rows
        neighbors.append(distances[x][0])
    return neighbors # ----------- > Here neighbours contains the whole row.


import operator
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]



def getAccuracy(testSet, predictions):
    correct = 0
    
    for x in range(len(testSet)):        
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def main():
    start = time.time()
    trainingSet=[]
    testSet=[] 
    split = 0.7 
    handleDataset('pulsar_stars -KNN.csv', split, trainingSet, testSet)     
    predictions=[] 
    k = 3
    
    for x in range(len(testSet)):
        neighbors = getKNeighbors(trainingSet, testSet[x], k) 
        result = getResponse(neighbors)  # --------- > M/B
        predictions.append(result)         
        
    accuracy = getAccuracy(testSet, predictions)    
    testSet_confusion_matrix = pd.DataFrame(testSet)
    testSet_confusion_matrix = testSet_confusion_matrix.iloc[:,-1]
    
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
    print('KNN ----- >')
    print('-------------------------------------------------------------')
    print('Confusion Matrix \n')
    print(cm)
    print('Accuracy: ' + repr(accuracy) + '%' + '\n')
    print(f'Sensitivity obtained for kNN : {Sensitivity} \n')
    print(f'Specificity obtained for kNN : {Specificity} \n')
    
    print(f'Precision : {Precision}')
    print(f'Recall : {Recall} \n')


    print(f'Total time taken to build KNN algorithm ------ > {end-start}s')

    
main()


# %%
import pandas as pd
import csv
import random
import time
import pandas as pd
from sklearn.metrics import confusion_matrix

def handleDataset(filename, split, trainingSet, testSet):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(8):
                dataset[x-1][y-1] = float(dataset[x-1][y-1])  
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

trainingSet=[]
testSet=[] 
split = 0.7 
                   
handleDataset('pulsar_stars -KNN.csv', split, trainingSet, testSet)