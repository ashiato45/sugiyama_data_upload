""" An assignment of ADA 6"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

np.random.seed(42)

H = 1
L = 1
EPS = 0.01

N = 200
X = np.hstack([np.random.randn(1, N//2) - 5, np.random.randn(1, N//2) + 5])
# add third line to simplify the calculation of f.
X = np.vstack([X, np.random.randn(1, N), np.ones((1, N))])
X = X.transpose()  # Make two groups
Y = np.vstack([np.ones((N//2, 1)), -np.ones((N//2, 1))])
X[0:3, 1] -= 5 # Make exceptional data
X[N//2:N//2+3, 1] += 5
Y[0:3] = -1
Y[N//2:N//2+3] = 1

def gk(x_, c_):
    return np.exp(-np.linalg.norm(x_ - c_)**2/(2*H**2))

def f(theta_, x_):
    v = np.matlib.repmat(x_.transpose(), N, 1) - X
    v = np.linalg.norm(v, axis=1)
    # v.shape must be (N, 1)
    v = v*v
    v /= -2*H**2
    v = np.exp(v)
    return np.dot(theta_.transpose(), v)


R = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        R[i, j] = gk(X[i, :], X[j, :])
# R = np.matlib.repmat(X.reshape((N, 1, 3)), 1, N)


theta = np.ones((N, 1))
while True:
    ind = np.zeros((N,))
    for i in range(N):
        ind[i] =  1 - f(theta, X[i])*Y[i, 0]
    d = np.fromfunction(lambda i, j: -Y[i, 0]*gk(X[i], X[j]), (N, N), dtype=int)
    d[ind<=0] = 0
    d = np.sum(d, axis=1).reshape((N, 1))
    d += L*np.dot(R, theta)
    theta2 = theta - EPS*d
    err = np.linalg.norm(theta2 - theta)
    if err < 10e-6:
        break
    print(err)
    theta = theta2

plt.axis([-10, 10, -10, 10])
plt.plot(X[(Y == 1).flatten(), 0], X[(Y == 1).flatten(), 1], 'bo')
plt.plot(X[(Y == -1).flatten(), 0], X[(Y == -1).flatten(), 1], 'rx')
plt.show()