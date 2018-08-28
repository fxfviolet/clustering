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

# 计算两个向量的欧氏距离
def distEclud(vecA,vecB):
    return np.sqrt(sum(np.power(vecA-vecB,2)))

# 密度聚类
def density_clustering(dataMat,minDistance,minNumber):
    m = np.shape(dataMat)[0]

    # 核心对象集合
    assemble = []
    for i in range(m):
        number = 0
        for j in range(m):
            distance = distEclud(dataMat[i],dataMat[j])
            if distance <= minDistance:
                number += 1
        if number >= minNumber:
            assemble.append(i)

    # 生成聚类簇
    cluster = {}
    k = 0
    while (len(assemble) > 0):
        cluster[k] = []
        rand_index = int(np.random.uniform(0,len(assemble)))   #
        for i in range(m):
            distance = distEclud(dataMat[assemble[rand_index]],dataMat[i])
            if distance <= minDistance:
                for j in range(m):
                    distance = distEclud(dataMat[i],dataMat[j])
                    if distance <= minDistance:
                        if i not in cluster[k]:
                            cluster[k].append(i)
        cluster[k] = set(cluster[k])
        if k > 0:
            for i in range(k):
                cluster[k] = cluster[k] -cluster[i]
        assemble = set(assemble)
        assemble = assemble - cluster[k]
        assemble = list(assemble)
    k += 1

    # 对聚类簇里的数据点求平均值，得到质心
    center=[]
    for key in cluster.keys():
        cluster[key] = list(cluster[key])
        mean_data = np.mean(dataMat[cluster[key]],axis=0).tolist()[0]
        center.append(mean_data)
    return center
  
dataMat = np.mat(load_data('test_data.txt'))
center = density_clustering(dataMat,minDistance=0.1,minNumber=5)