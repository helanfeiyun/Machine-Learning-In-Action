from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group , labels


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


group, labels = createDataSet()
result = classify0([0,1],group,labels,3)
print(result)


#程序清单2-2 讲文本中数据转换成矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines ,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
print(datingDataMat,datingLabels)

fig = plt.figure()
ax = fig.add_subplot(212)
ax.scatter(datingDataMat[:,0],datingDataMat[:,2],s=15*array(datingLabels), c=array(datingLabels))
plt.show()

#程序清单2-3 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(maxVals,(m,1))
    return normDataSet, ranges, minVals

normDataSet,ranges,minVals = autoNorm(datingDataMat)
print(normDataSet,ranges,minVals)

def datingClassTest():
    hoRatio  = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    print( m )
    numTestVecs = 2
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with : %d, the real answer is : %d" %(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is : %f" %(errorCount/float(numTestVecs)))

datingClassTest()