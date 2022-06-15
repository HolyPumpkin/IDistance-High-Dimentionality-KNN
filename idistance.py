# coding=utf-8
"""
IDistance实现核心算法
--By Holypumpkin
"""
import datetime

import numpy as np
import pandas as pd
import kmeans
import sort_data

start = datetime.datetime.now()
query_index_point = pd.read_csv('query_index_point.csv', index_col=0, sep=',')
query_index_dist = pd.read_csv('query_index_dist.csv', index_col=0, sep=',')
full_data = pd.read_csv('corel.csv', sep=' ', index_col=0, header=None)
part_map = pd.read_csv('partition_res.csv', sep=',', index_col=0, )
final_center = pd.read_csv('final_center.csv', sep=',', index_col=0)
normal_search = pd.read_csv('normal_search.csv', index_col=0, sep=',')

l1 = query_index_point.values.tolist()
l2 = query_index_dist.values.tolist()
l3 = full_data.values.tolist()
l4 = part_map.values.tolist()
l5 = final_center.values.tolist()
l6 = normal_search.values.tolist()

query_index_point = np.array(l1)
query_index_dist = np.array(l2)
full_data = np.array(l3)
part_map = np.array(l4)
final_center = np.array(l5)
normal_search = np.array(l6)

valid_len = []
for _ in range(20):
    valid_len.append(len(query_index_point[_]) - np.isnan(query_index_point[_]).sum())


def get_centers_radius():
    """
    得到所有聚类的最大半径
    这里以聚类中心作为reference point
    """
    result = []
    for i in range(20):
        max_index = query_index_point[i][valid_len[i] - 1]
        result.append(kmeans.distance(full_data[int(max_index)], final_center[i]))
    return result


def get_dist_to_center(p_id):
    """
    得到查询点到所有聚类中心的距离
    """
    result = []
    for i in range(20):
        result.append(kmeans.distance(full_data[p_id], final_center[i]))
    return result


def get_center(p_id):
    """
    [parameter]
    p_id:查询点的id号
    [return]
    查询点所属聚类中心的序号
    """
    return part_map[p_id][0]


def first_binary_search(center_id, left_r, right_r):
    """
    [parameters]
    center_id:要查找的聚类中心编号
    left_r:查找左边界
    right_r:查找右边界
    第一次二分查找，只是寻找到一个符合条件的点
    因为index是按距离升序存储的，所以找到一个后，从中心向两边扩散即可找到其他点
    """
    left = 0
    right = len(query_index_dist[center_id]) - 1
    while left != right:
        mid = int((left + right) / 2)
        # print(mid)
        if right_r >= query_index_dist[center_id][mid] >= left_r:
            # print("it's time to return")
            return mid
        elif right_r < query_index_dist[center_id][mid]:
            right = mid - 1
            # print("right = mid")
        else:
            # print("left = mid")
            left = mid + 1
    print("范围内无符合要求的点，请扩大半径！")
    return -1


def binary_search(left_r, right_r, center_id, mid, k):
    """
    [parameters]
    left_r:查找左边界
    right_r:查找右边界
    center_id:查找聚类中心编号
    mid:第一次找到的中心点，由此向两边扩散，搜寻范围内有效点
    """
    result = [query_index_point[center_id][mid]]
    left = mid - 1
    nums = 0
    while left >= 0 and query_index_dist[center_id][left] >= left_r:
        result.append(query_index_point[center_id][left])
        nums += 1
        if nums >= k:
            return result, nums
    right = mid + 1
    while right < len(query_index_dist[center_id]) and query_index_dist[center_id][right] <= right_r:
        result.append(query_index_point[center_id][right])
        nums += 1
        if nums >= k:
            return result, nums
    return result, nums


def i_distance(point_id, delt_r, k):
    """
    [parameters]
    point_id:查询点的id号
    delt_r:半径增长值，初始半径与此相同
    K:k个近邻
    """
    radius = get_centers_radius()  # 得到每个聚类的最大半径
    dist_to_center = get_dist_to_center(point_id)  # 得到查询点与每个聚类参考点的距离
    contain_center = get_center(point_id)  # 得到查询点所属的聚类中心编号
    r = delt_r
    left = dist_to_center[contain_center] - r
    right = min(radius[contain_center], dist_to_center[contain_center] + r)
    mid = first_binary_search(contain_center, left, right)
    result, num = binary_search(left, right, contain_center, mid, k)
    while r <= radius[contain_center]:
        need_query = [contain_center]
        query_range = [[dist_to_center[contain_center] - r, dist_to_center[contain_center] + r]]
        for i in range(20):  # 若查找半径+该聚类半径大于查找点与该聚类中心的距离，
            # 说明查找半径与该聚类相交，则也需要查询此聚类
            if (i not in need_query) and (r + radius[i] > dist_to_center[i]):
                need_query.append(i)  # 更新需要查找的聚类中心即查找范围,范围即为两圆交集在此聚类中的扩展区域
                query_range.append([radius[i] - (dist_to_center[i] - r), radius[i]])
        for i in range(len(need_query)):
            mid = first_binary_search(need_query[i], query_range[i][0], query_range[i][1])
            if mid == -1:
                continue
            t, num = binary_search(query_range[i][0], query_range[i][1], i, mid, k)
            result += t
            result = list(set(result))
            if len(result) - np.isnan(np.array(result)).sum() >= k * 10:
                return result
        r += delt_r

    return result


def main():
    data = []
    for c in range(1000):
        result = i_distance(c, 0.1, 10)
        n = np.array(result)
        df = pd.DataFrame(n)
        df = df.dropna()
        a = np.array(df)
        row = []
        for i in a:
            row.append(i[0])
        data.append(row)
    end = datetime.datetime.now()
    data = sort_data.sort(data)
    accuracy_mat = []
    for i in range(1000):
        count = 0
        for j in range(10):
            if normal_search[i][j] == data[i][j]:
                count += 1
        accuracy_mat.append(count/float(10))
    avg_accuracy = sum(accuracy_mat)/1000
    print("平均准确率为:")
    print(avg_accuracy)
    print("查询1000个点所需时间为:")
    print(end-start)
    df2 = pd.DataFrame(data)
    df2.to_csv('idistance_result.csv')


if __name__ == '__main__':
    main()
