# Hardik Sahi
# University of Waterloo
# Classification: Implementation of Winnow algorithm

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


def checkSign(inputRow, outputVal, bias,weightArray):
    dot = np.dot(inputRow,weightArray)
    dot = outputVal*(dot + bias)
    return dot
    
def updateWt(inputRow, outputVal, lRate,weightArray):
    inputRow = inputRow*lRate*outputVal
    return np.multiply(weightArray,np.exp(inputRow))

    
def winnow(inputArray, outputArray,lRate,maxN):
    dimensionN = inputArray.shape[1] #number of dimensions per training set input
    w = np.ones(dimensionN,float) # 1-d array (d,)
    bias =  float(1/(dimensionN+1))
    w = bias*w
    mistake = np.zeros(maxN,int)
    nDataPoints = inputArray.shape[0]
    for t in range(0,maxN):
        mistake[t] = 0
        for i in range(0,nDataPoints):
            if checkSign(inputArray[i],outputArray[i,0],bias,w)<=0:
                w = updateWt(inputArray[i],outputArray[i,0],lRate,w)
                bias *=np.exp(lRate*outputArray[i,0])              
                normalizeF = float(bias + np.sum(w))
                w = w/normalizeF
                bias = bias/normalizeF
                mistake[t]+=1

    return mistake,w,bias

mistakeArray,weightArray,bias = winnow(inputArray,outputArray,(0.00001),500)
import matplotlib.pyplot as plt
plt.plot(range(0,len(mistakeArray)),mistakeArray)
plt.xlabel("Pass number")
plt.ylabel("Mistake count for eta: %f" %(0.00001))
plt.show()



