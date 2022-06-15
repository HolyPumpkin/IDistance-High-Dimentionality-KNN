# coding=utf-8
import random
from collections import defaultdict
from random import uniform
from math import sqrt

import numpy as np


def point_avg(points):
    """
    求取质心函数
    输入一个以点为元素的列表，其中每个点都是同样的维度（本题中为32维）
    返回一个质心点（平均值点）
    """
    dimensions = len(points[0])

    new_center = []

    for dimension in xrange(dimensions):
        dim_sum = 0  # 每一维度的value和
        for p in points:
            dim_sum += p[dimension]

        # 每一维度的平均value值
        new_center.append(dim_sum / float(len(points)))

    return new_center


def update_centers(data_set, assignments):
    """
    更新聚类质心函数
    输入数据集以及一个指派列表，指派列表的下标与数据集每个点对应
    指派assignments[i][j]即表示 i 号点在 j 号聚类中。
    返回k个更新好的聚类质心
    """
    new_means = defaultdict(list)  # 创建字典，键为k个聚类质心，值为属于此聚类的点
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for points in new_means.itervalues():
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
    """
    指派函数，将数据集中的每个点划分入最近的聚类中
    输入数据集以及聚类质心列表
    返回一个指派列表，其元素总数与数据集一致，对应每个点
    元素为质心索引，表示该点被指派到对应聚类
    """
    assignments = []
    for point in data_points:
        shortest = ()  # 使得第一次比较总为真，相当于正无穷
        shortest_index = 0
        for i in xrange(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    得到a、b两点的欧氏距离
    """
    dimensions = len(a)

    _sum = 0
    for dimension in xrange(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)


def cal_distort(old_centers, new_centers):
    """
    计算前后两个聚类质心抖动的函数
    返回平均抖动
    """
    distort = 0
    k = len(new_centers)
    for i in xrange(k):
        distort += distance(old_centers[i], new_centers[i])
    distort = distort / float(k)

    return distort


def generate_k(data_set, k):
    """
    对输入数据集，随机生成k个初始质心
    返回随机生成的初始质心
    """
    centers = []
    l1 = random.sample(range(0, len(data_set)), k)
    for i in l1:
        centers.append(data_set[i])

    return centers


def k_means(dataset, k, thresh=1e-3, count=20):
    """
    K-means核心部分
    输入分别为：
    dataset:数据集
    k:期望聚类数
    thresh:抖动阈值，当聚类的两次质心平均抖动小于该值时，一次kmeans结束
    count:迭代次数
    返回为 最终指派与索引、最终质心列表
    """
    k_points = generate_k(dataset, k)   # 生成k个随机初始质心
    assignments = assign_points(dataset, k_points)  # 生成初始的指派列表
    old_assignments = None
    distort = np.inf    # 抖动初值为无穷
    old_centers = k_points
    num = count
    while assignments != old_assignments and distort > thresh and num >= 0:
        new_centers = update_centers(dataset, assignments)  # 更新质心
        distort = cal_distort(old_centers, new_centers) # 计算抖动
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)   # 获得新指派
        old_centers = new_centers
        num -= 1
    dataset_index = [i for i in xrange(len(dataset))]
    return zip(assignments, dataset_index), old_centers


'''
以下为测试程序
'''
# points = [
#     [1, 2],
#     [2, 1],
#     [3, 1],
#     [5, 4],
#     [5, 5],
#     [6, 5],
#     [10, 8],
#     [7, 9],
#     [11, 5],
#     [14, 9],
#     [14, 14],
# ]
# a, b = k_means(points, 4)
# print a
# print b
# out = np.array(a)
# np.save('out.npy', out)
# data_in = np.load('out.npy')
# print data_in
