# coding=utf-8
import numpy as np
import pandas as pd
from math import sqrt

'''
普通查询，即暴力计算1000个待查询点与其余点的距离
之后遍历寻找k近邻（k=10）
'''


def distance(a, b):
    dimensions = len(a)
    _sum = 0
    for dimension in xrange(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)


full_data = pd.read_csv('corel.csv', index_col=0, sep=' ', header=None)
l1 = full_data.values.tolist()
full_data = np.array(l1)

given_data = pd.read_csv('corel.csv', index_col=0, sep=' ', nrows=1000, header=None)
l2 = given_data.values.tolist()
given_data = np.array(l2)

dist_mat = []
for i in given_data:
    dist_vector = []
    for j in full_data:
        dist_vector.append(distance(i, j))
    dist_mat.append(dist_vector)

result = []
for i in range(1000):
    match = []
    b = sorted(enumerate(dist_mat[i]), key=lambda x: x[1])
    for j in range(10):
        match.append(b[j][0])
    result.append(match)

df = pd.DataFrame(result)
df.to_csv('normal_search.csv')
