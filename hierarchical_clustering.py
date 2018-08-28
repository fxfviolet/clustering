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

# 计算欧氏距离
def distEclud(vecA,vecB):
    return np.sqrt(sum(np.power(vecA-vecB,2)))

# 计算两个聚类簇的平均距离
def distAvg(Ci,Cj):
    return np.sum(distEclud(i,j) for i in Ci for j in Cj)/(len(Ci)*len(Cj))

# 找出距离最小的下标
def find_min_distance(distance):
    min_distance = np.inf
    x=-1; y=-1 
    for i in range(len(distance)):
        for j in range(len(distance[0])):
            if i != j and distance[i,j] < min_distance:
                min_distance = distance[i,j]
                x=i; y=j 
    return x,y,min_distance

# 层次聚类
def hierarchical_cluster(dataMat,k):
    # 初始化聚类簇，每个样本作为一个聚类簇
    cluster = []
    for i in range(len(dataMat)):
        Ci = []
        Ci.append(i)
        cluster.append(Ci)

    # 计算所有点之间的距离
    distance = []
    for i in range(len(cluster)):
        Di = []
        for j in range(len(cluster)):
            Di.append(distAvg(dataMat[cluster[i][0]],dataMat[cluster[j][0]]))
        distance.append(Di)
    distance = np.mat(distance)

    # 合并更新簇
    m = len(dataMat)
    while m > k:
        x,y,min_distance = find_min_distance(distance)
        cluster[x].extend(cluster[y])         # 距离最近的两个簇合并
        cluster.remove(cluster[y])            # 删除原有的、已被合并的簇

        # 计算新的簇之间的距离
        distance = []
        for i in range(len(cluster)):
            Di = []
            for j in range(len(cluster)):
                Di.append(distAvg(dataMat[cluster[i][0]],dataMat[cluster[j][0]]))
            distance.append(Di)
        distance = np.mat(distance)
        m -= 1

    # 计算质心
    center = []
    for i in range(k):
        center.append(np.mean(dataMat[cluster[i]],axis=0))
    return center

dataMat = np.mat(load_data('test_data.txt'))
center = hierarchical_cluster(dataMat,k)
