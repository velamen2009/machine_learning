# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:19:25 2016

@author: lmei
"""

import numpy as np
import matplotlib.pyplot as plt

def loadSimpData():
    dataMat = np.matrix([[1.0, 2.1], [2.0, 1.1], [1.3, 1.0], [1.0, 1.0], [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def plotSimpData(dataMat, classLabels):
    f=plt.figure(1)
    colors = {1:'r', -1:'g'}
    for i in range(len(dataMat)):
        plt.scatter(dataMat[i][0, 0], dataMat[i][0, 1], color = colors[classLabels[i]], marker = 'o', s=100)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Simple Data')
    plt.legend()
    f.show()
    
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray
    
def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = np.dot(D.T, errArr)
                #print('split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f' % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst
    
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print('D: ', D.T)
        alpha = float(0.5*np.log((1.0 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst: ', classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, np.mat(classEst))
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print('aggClassEst: ', aggClassEst.T)
        aggErrors = np.dot(np.sign(aggClassEst).T != np.mat(classLabels), np.ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print('total error: ', errorRate, '\n')
        if errorRate == 0.0: break
    return weakClassArr
            
def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)
    
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    with open(fileName, 'r') as fr:
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat-1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
    return dataMat, labelMat
            


if __name__ == '__main__':
    dataMat, classLabels = loadSimpData()
    plotSimpData(dataMat, classLabels)
    D = np.mat(np.ones((5, 1))/5)
    #bestStump, minError, bestClassEst = buildStump(dataMat, classLabels, D)
    classifierArray = adaBoostTrainDS(dataMat, classLabels, 9)
    print('[0, 0]', adaClassify([0, 0], classifierArray))
    print('[5, 5]', adaClassify([5, 5], classifierArray))
    print('[10, 10]', adaClassify([10, 10], classifierArray))
    
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    horseClassifierArr = adaBoostTrainDS(dataArr, labelArr, 10)
    
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction10 = adaClassify(testArr, horseClassifierArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    errCount = errArr[prediction10 != np.mat(testLabelArr).T].sum()
    errRate = errCount/float(len(testLabelArr))
    print('%d of %d were errors, error rate is: %.3f' % (errCount, len(testArr), errRate))
    