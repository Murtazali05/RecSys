import numpy as np
from sklearn import decomposition as d

d._truncated_svd()
matrix = [[1, 2, 3],
          [5, 4, 1]]

if matrix != 2:
    print('Matrix')

for i in enumerate(matrix):
    print(i)

print()

sim_scores = list(enumerate(matrix))
print(sim_scores)
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

print(sim_scores)

arr = np.array([1, 2, 4])
print(arr)


tmp = [1]

tmp.append()



