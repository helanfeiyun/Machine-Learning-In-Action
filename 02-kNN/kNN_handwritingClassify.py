from numpy import *
from os import listdir
import operator

def classify0(inX, dataSet, labels , k):
    #shape 用来查看矩阵或者数组的维数
    #tile（A,（B1,B2）) ： 重建数组1维重复A,B1次，2维重复A,B2次
    #sum(x): x为空,所有数相加; x=0, 每行对应列相加; x=1, 每行自己相加
    #argsort(a,axis= ): 返回数组值从小到大的索引值
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(linStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('MLiA_SourceCode/machinelearninginaction/Ch02/digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('MLiA_SourceCode/machinelearninginaction/Ch02/digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('MLiA_SourceCode/machinelearninginaction/Ch02/digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('MLiA_SourceCode/machinelearninginaction/Ch02/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat, hwLabels,3)
        print("the classifier came back with: %d, the real answer is : %d" %(classifierResult,classNumStr))
        if( classifierResult != classNumStr):
            errorCount += 1.0
    print("\n the total number of errors is :%d." % errorCount)
    print("\n the total error rate is :%f." % (errorCount/float(mTest)))

handwritingClassTest()