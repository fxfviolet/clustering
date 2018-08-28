from numpy import *

# 加载文件，读取数据
def load_data(fileName):
    dataSet = []
    openfile = open(fileName)
    for line in openfile.readlines():
        curLine = line.strip().split('\t')
        floatLine = map(float,curLine)
        dataSet.append(floatLine)
    return dataSet

# 计算两个向量的欧氏距离
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

# 随机选取初始质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

# k-means均值算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0) 
    return centroids, clusterAssment

# 二分k-means均值算法
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]     # 平均值作为初始质心
    centList =[centroid0]                             # 创建质心列表
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)   # 调用k-means求质心
            sseSplit = sum(splitClustAss[:,1])                                   # 质心到数据点的距离求和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            if (sseSplit + sseNotSplit) < lowestSSE:                             # 如果距离比原来的距离小，更新质心
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)     # 索引号更改为质心列表的长度
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit

        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]                # 更新为新的质心
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss  # 更新为新的簇
    return centList, clusterAssment

dataSet = mat(load_data('test_data.txt'))
centList,clusterAssment = biKmeans(dataSet,k,distMeas=distEclud)

