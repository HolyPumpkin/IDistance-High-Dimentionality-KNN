import numpy as np
import pandas as pd
import kmeans

full_data = pd.read_csv('corel.csv', sep=' ', index_col=0, header=None)
part_data = pd.read_csv('partition_res.csv', sep=',', index_col=0)
part_centers = pd.read_csv('final_center.csv', sep=',', index_col=0)

full_data = np.array(full_data)
part_data = np.array(part_data)
part_centers = np.array(part_centers)

nums = len(full_data)
point_index = [[] for _ in range(20)]
dist_index = [[] for _ in range(20)]

for i in range(nums):
    order = part_data[i][0]
    dist = kmeans.distance(full_data[i], part_centers[order])
    point_index[order].append(i)
    dist_index[order].append(dist)

result = []
for j in range(20):
    arr = list(zip(point_index[j], dist_index[j]))
    arr.sort(key=lambda x: (x[1], x[0]))
    result.append(arr)

for i in range(20):
    for j in range(len(point_index[i])):
        point_index[i][j] = result[i][j][0]
        dist_index[i][j] = result[i][j][1]


# print point_index
# print dist_index

df = pd.DataFrame(point_index)
df.to_csv('query_index_point.csv')

df2 = pd.DataFrame(dist_index)
df2.to_csv('query_index_dist.csv')

