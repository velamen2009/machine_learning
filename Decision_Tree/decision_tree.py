# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 09:34:01 2016

@author: lmei
"""

from math import log
import operator
import matplotlib.pyplot as plt
import os

filename = 'decision_tree'

def calcShannonEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    ShannonEntropy = 0.0
    for key in labelCounts.keys():
        prob = float(labelCounts[key]/numEntries)
        ShannonEntropy -= prob*log(prob, 2)
    return ShannonEntropy

def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduced = featVec[:axis]
            reduced.extend(featVec[axis+1:])
            retDataSet.append(reduced)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEntropy(subDataSet)
        InfoGain = baseEntropy - newEntropy
        if InfoGain > bestInfoGain:
            bestInfoGain = InfoGain
            bestFeature = i
    return bestFeature
    
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    print(classList)
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
        
# plot
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', 
                            xytext=centerPt, textcoords='axes fraction', 
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)
def createPlot_test():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree)[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree)[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees = [{'no surfacing':{0:'no', 1:{'flippers':{0:'no', 1:'yes'}}}},
                   {'no surfacing':{0:'no', 1:{'flippers':{0:{'head':{0:'no', 1:'yes'}}, 1:'no'}}}} 
    ]
    return listOfTrees[i]

def plotMidText(cntrPt, parentPt,  txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)
    
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree)[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText (cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
    
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5,1.0), ' ')
    plt.show()

#classify
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree)[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel
    
#storage
def storeTree(inputTree):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(inputTree, f)
    
def grabTree():
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)
        
class DecisionTree:
    def __init__(self, labels, tree):
        self.labels = labels
        self.tree = tree
    
if __name__ == '__main__':
    if not os.path.exists(filename):
        print('Builing the tree...')
        myDat, labels = createDataSet()
        entropy = calcShannonEntropy(myDat)
        print('The input dataset:', myDat)
        classifyLabel = labels[:]
        print('The input labels:', labels)
        print('The base Shannon entropy:', entropy)
        myTree = createTree(myDat, classifyLabel)
        print('The decision tree:', myTree)
        leafs = getNumLeafs(myTree)
        depth = getTreeDepth(myTree)
        print('The number of leafs of retrieved tree:', leafs)
        print('The depth of retrieved tree:', depth)
        decisionTree = DecisionTree(labels, myTree)
        storeTree(decisionTree)
    else:
        decisionTree = grabTree()
        labels = decisionTree.labels
        myTree = decisionTree.tree
    #testTree = retrieveTree(0)
    #print('The retrieved tree:', testTree)
    #createPlot_test()
    createPlot(myTree)
    classLabel = classify(myTree, labels, [0,0])
    print("[0, 0]", classLabel)
    classLabel = classify(myTree, labels, [0,1])
    print("[0, 1]", classLabel)
    classLabel = classify(myTree, labels, [1,0])
    print("[1, 0]", classLabel)
    classLabel = classify(myTree, labels, [1,1])
    print("[1, 1]", classLabel)