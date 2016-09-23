# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 09:51:28 2016

@author: lmei
"""

import numpy as np
import matplotlib.pyplot as plt
import random

def loadDataSet():
    dataMat = []
    labelMat = []
    with open('data/testSet.txt', 'r') as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat
    
def sigMoid(inX):
    return 1.0/(1 + np.exp(-inX))

# gradient ascent
# w:= w + alpha*delta(y)*x   
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigMoid(np.dot(dataMatrix, weights))
        error = labelMat - h
        weights = weights + alpha*np.dot(dataMatrix.transpose(), error)
    return weights

#
def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigMoid(sum(np.array(dataMatrix[i])*weights))
        error = classLabels[i] - h
        weights = np.add(weights, alpha*error*np.array(dataMatrix[i]))
    return weights
    
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 +j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigMoid(sum(np.array(dataMatrix[randIndex])*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha*error*np.array(dataMatrix[randIndex])
            del(dataIndex[randIndex])
    return weights
    
def plotBestFit(wei):
    #weights = wei.getA()
    weights = wei
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='r', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='b', marker='o')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def classifyVector(inX, weights):
    prob = sigMoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('data/horseColicTraining.txt'); frTest = open('data/horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate
            
        
        
if __name__ == '__main__':
    dataMatIn, classLabels = loadDataSet()
    wei = gradAscent(dataMatIn, classLabels)
    plotBestFit(wei.getA())
    wei0 = stocGradAscent0(dataMatIn, classLabels)
    plotBestFit(wei0)
    wei1 = stocGradAscent1(dataMatIn, classLabels)
    plotBestFit(wei1)
    colicTest()