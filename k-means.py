import numpy as np

# 加载文件，读取数据
def load_data(fileName):
    dataMat = []
    openfile = open(fileName)
    for line in openfile.readlines():
        curLine = line.strip().split('\t')
        floatLine = map(float,curLine)
        dataMat.append(floatLine)
    return dataMat

# 随机选取初始质心
def randCent(dataMat, k):
    n =np.shape(dataMat)[1]
    centroids =np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataMat[:,j])                  # 找出矩阵dataMat第j列最小值
        rangeJ = float(max(dataMat[:,j]) - minJ)  # 计算第j列最大值和最小值的差
        centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)  # 赋予一个随机质心，它的值在整个数据集的边界之内
    return centroids

# 计算两个向量的欧氏距离
def distEclud(vecA,vecB):
    return np.sqrt(sum(np.power(vecA-vecB,2)))

# k-means均值算法
def kMeans(dataMat,k,distE = distEclud , createCent=randCent):
    m = np.shape(dataMat)[0]
    clusterAssment = np.mat(np.zeros((m,2))) # 初试化一个矩阵，用来记录簇索引和存储误差
    centroids = createCent(dataMat,k)        # 随机的得到一个质心矩阵蔟
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):                                   #对每个数据点寻找最近的质心
            minDist = np.inf; minIndex = -1
            for j in range(k):                              # 遍历质心蔟，寻找最近的质心
                distJ1 = distE(centroids[j,:],dataMat[i,:])  # 计算数据点和质心的欧氏距离
                if distJ1 < minDist:
                    minDist = distJ1; minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        for cent in range(k):                               # 更新质心的位置
            ptsInClust = dataMat[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment

dataMat = np.mat(load_data('test_data.txt'))
centroids,clusterAssment = kMeans(dataMat,k)
