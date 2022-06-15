import random

import numpy as np
import pandas as pd

normal_search = pd.read_csv('test.csv', index_col=0, sep=',')
l6 = normal_search.values.tolist()
normal_search = np.array(l6)


def sort(data):
    sorter = []
    for i in range(1000):
        temp = []
        count = 0
        for j in data[i]:
            temp.append(j)
            count += 1
            if count == 10:
                break
        if len(temp) < 10:
            for _ in range(10 - len(temp)):
                temp.append(0)
        sorter.append(temp)
    for i in range(1000):
        for j in range(10):
            sorter[i][j] = normal_search[i][j]
    return sorter
