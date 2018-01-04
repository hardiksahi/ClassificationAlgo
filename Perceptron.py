# Hardik Sahi
# University of Waterloo
# Classification: Implementation of Perceptron algorithm 

import numpy as np
import csv

inputfile = open('spambase_X.csv', 'r')
inputreader = csv.reader(inputfile,delimiter=',')
inputData = list(inputreader)
inputArray = np.array(inputData).astype("float")

outputFile = open('spambase_y.csv', 'r')
outputreader = csv.reader(outputFile, delimiter = ',')
outputData = list(outputreader)
outputArray = np.array(outputData).astype("float")

print("xTrainInput siz", inputArray.shape)
print("yTainINput size", outputArray.shape)

dataPoints = inputArray.shape[0]
dimensions = inputArray.shape[1]


def dotProd(inputRow, output, bias, weightArray):
    dot = np.dot(inputRow,weightArray) +bias
    return output*dot



def perceptron(inputArray, outputArray, maxTries):
    weightArray = np.zeros(inputArray.shape[1], float)
    bias = 0
    mistakeArray = np.zeros(maxTries,int)
    for t in range(maxTries):
        mistakeArray[t] = 0
        for i in range(inputArray.shape[0]):
            if dotProd(inputArray[i],outputArray[i,0],bias,weightArray) <=0:
                weightArray = np.add(weightArray,outputArray[i,0]*inputArray[i])
                bias = bias + outputArray[i,0]
                mistakeArray[t]+=1
                
    return mistakeArray            
    
mistakeArr = perceptron(inputArray,outputArray,10)
print(mistakeArr)
print("mistakePercent", np.sum(mistakeArr)/(10*dataPoints))
import matplotlib.pyplot as plt
plt.plot(range(0,len(mistakeArr)),mistakeArr)
plt.xlabel("Pass number")
plt.ylabel("Mistake count")
plt.show()
















