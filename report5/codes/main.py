import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import gc

np.random.seed(42)

digitmat = scipy.io.loadmat("digit.mat")
X = digitmat["X"]
T = digitmat["T"]
X = np.hstack([X[:, :, i] for i in range(10)])
T = np.hstack([T[:, :, i] for i in range(10)])
tpoints = 5000
del digitmat
gc.collect()

print(T.shape, X.shape)


# learn!

h = 1
lam = 1
# _K1 = np.zeros((5000, 5000))
# for i in range(5000):
#     for j in range(5000):
#         _K1[i, j] = np.linalg.norm(X[:, i] - X[:, j])
#     print(i)
# # _K1 = np.fromfunction(lambda a, b: np.linalg.norm(X[:, a] - X[:, b]), (tpoints, tpoints), dtype=int)
# print(_K1.shape)
# np.save("k1.npy", _K1)
_K1 = np.load("k1.npy")
K = np.exp((-1)*np.square(_K1)/(2*h*h))

anss = np.ones((5000, 10))
anss = -anss
for i in range(10):
    for j in range(500*i, 500*(i+1)):
        anss[j, i] = 1
theta = np.linalg.solve(np.dot(K, K) + lam*np.eye(5000), np.dot(K.transpose(), anss))
print(theta.shape)

# Classify!

# _KK1 = np.zeros((2000, 5000))
# for i in range(2000):
#     for j in range(5000):
#         _KK1[i, j] = np.linalg.norm(T[:, i] - X[:, j])
#     print(i)
# print(_KK1.shape)
# np.save("kk1.npy", _KK1)
_KK1 = np.load("kk1.npy")
KK = np.exp((-1)*np.square(_KK1)/(2*h*h))

ys = np.dot(KK, theta)
print(ys[0:10,:])

# Check!
cla = np.floor(np.argmax(ys, axis=1))
print(cla, cla.size)
correct = np.floor(np.arange(2000)/200)
print(correct, correct.size)
print(cla == correct)
print((cla == correct).size)
