import numpy as np
import pandas as pd
import kmeans

query_index_point = pd.read_csv('query_index_point.csv', index_col=0, sep=',')
query_index_dist = pd.read_csv('query_index_dist.csv', index_col=0, sep=',')
full_data = pd.read_csv('corel.csv', sep=' ', index_col=0, header=None)
part_map = pd.read_csv('partition_res.csv', sep=',', index_col=0, )
final_center = pd.read_csv('final_center.csv', sep=',', index_col=0)
l1 = query_index_point.values.tolist()
l2 = query_index_dist.values.tolist()
l3 = full_data.values.tolist()
l4 = part_map.values.tolist()
l5 = final_center.values.tolist()

query_index_point = np.array(l1)
query_index_dist = np.array(l2)
full_data = np.array(l3)
part_map = np.array(l4)
final_center = np.array(l5)
#
# valid_len = []
# for _ in range(20):
#     valid_len.append(len(query_index_point[_]) - np.isnan(query_index_point[_]).sum())
#
# print(valid_len)
# for i in range(20):
#     print(query_index_point[i])
#     print(query_index_point[i][valid_len[i]-1])

left = 0
right = 6000
q = 13000
mid = (left + right) / 2
print mid
while left < right:
    mid = int((left + right) / 2)
    print(mid)
    if right >= q >= left:
        print("it's time to return")
        break
    elif q < left:
        right = mid-1
        print("right = mid")
    else:
        print("left = mid")
        left = mid+1
