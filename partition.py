import numpy as np

import kmeans
import pandas as pd

file_data = pd.read_csv('corel.csv', index_col=0, sep=' ', header=None)
l1 = file_data.values.tolist()
data = np.array(l1)

print(data)
print(data.shape)

part_result, final_centers = kmeans.k_means(data, 20)
# print(part_result)
print(final_centers)
df = pd.DataFrame(part_result)
df.to_csv('partition_res.csv')
df_center = pd.DataFrame(final_centers)
df_center.to_csv('final_center.csv')
