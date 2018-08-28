import numpy as np
from math import *

# 加载文件，读取数据
def load_data(fileName):
    dataMat = []
    openfile = open(fileName)
    for line in openfile.readlines():
        curLine = line.strip().split('\t')
        floatLine = map(float,curLine)
        dataMat.append(floatLine)
    return dataMat

# 概率密度函数
def probability_density(data,center,covariance,n):
    expon = np.exp((-0.5) * (data - center) * covariance.I * (data - center).T)
    denom = (2 * pi) ** (n/2) * np.sqrt(np.linalg.det(covariance))
    return expon/denom

# 初始化质心、混合系数、协方差矩阵
def initial_center(dataMat, k):
    n = np.shape(dataMat)[1]
    center = np.mat(np.zeros((k,n)))                # 质心
    for j in range(n):
        minJ = min(dataMat[:,j])
        rangeJ = float(max(dataMat[:,j]) - minJ)
        center[:,j] = minJ + rangeJ * np.random.rand(k,1)
    coefficient = np.random.rand(k)                # 混合系数
    coefficient /= sum(coefficient)
    covariance = []                                # 协方差矩阵
    for i in range(k):
        covariance.append(np.mat([[0.1,0.0],[0.0,0.1]]))
    return center,coefficient,covariance

# 高斯混合聚类
def mixture_gaussian(dataMat,k):
    m = np.shape(dataMat)[0]
    probability = np.mat(np.zeros((m,k)))   # 各混合成分的后验概率
    center,coefficient,covariance = initial_center(dataMat, k)
    error = 0.001
    covarinace_nonzero = True
    while covarinace_nonzero:
        center_error = 0
        coefficient_error = 0
        init_center = center.copy()
        init_coefficient = coefficient.copy()
        init_covariance = covariance.copy()

        # 计算各混合成分的后验概率
        for j in range(m):
            init_data = dataMat[j]
            denom = 0
            for i in range(k):
                denom += init_coefficient[i] * probability_density(init_data,init_center[i],init_covariance[i],k)
            for i in range(k):
                numer = init_coefficient[i] * probability_density(init_data,init_center[i],init_covariance[i],k)
                probability[j,i] = float(numer/denom)

        # 更新模型参数
        for i in range(k):
            center_numer = 0
            center_denom = 0
            for j in range(m):
                center_numer += probability[j,i] * dataMat[j]
                center_denom += probability[j,i]
            center[i] = center_numer / center_denom
            coefficient[i] = np.sum(probability[:,i])/m
            covariance_numer = np.mat(np.zeros((2,2)))
            covariance_denom = 0
            for j in range(m):
                covariance_numer += probability[j,i] * (dataMat[j] - center[i]).T * (dataMat[j] - center[i])
                covariance_denom += probability[j,i]
            covariance[i] = covariance_numer/covariance_denom

        # 判断协方差矩阵是否是奇异值
        for i in range(k):
            if np.linalg.det(covariance[i]) == 0:
                covarinace_nonzero = False
                break

        # 计算误差
        for i in range(k):
            center_error += np.sum(abs(init_center[i] - center[i]))
            coefficient_error += abs(init_coefficient[i] - coefficient[i])
        if center_error < error and coefficient_error < error:
            break

    return center

dataMat = np.mat(load_data('test_data.txt'))
center = mixture_gaussian(dataMat,k)


