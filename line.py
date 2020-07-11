import numpy as np
import itertools


a = [1, 1]
b = [2, 2]
aa = [3, 3]
bb = [4, 4]

s = np.vstack([a, b, aa, bb])
h = np.hstack((s, np.ones((4, 1))))
print(h)

print(np.cross(h[0], h[1]))

arr = [1, 2]
print(list(itertools.permutations(arr)))
