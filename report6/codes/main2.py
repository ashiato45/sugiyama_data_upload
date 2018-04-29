""" An assignment of ADA 6"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

np.random.seed(42)

H = 1
L = 1
EPS = 10e-6

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

def f(theta_, x_):
    return np.dot(theta_.transpose(), x_)


theta = np.ones((3, 1))
while True:
    d = -X*np.matlib.repmat(Y, 1, 3)  # Nx3
    ind = 1 - Y*X.dot(theta)
    d[ind <= 0] = 0
    d = np.sum(d, axis=0) #1x3
    d = d.reshape(3, 1) #3x1
    d += L*theta
    theta2 = theta - EPS*d
    err = np.linalg.norm(theta2 - theta)
    print(err)
    theta = theta2
    if err < 10e-5:
        break


print(theta)        


plt.axis([-10, 10, -10, 10])
plt.plot(X[(Y == 1).flatten(), 0], X[(Y == 1).flatten(), 1], 'bo')
plt.plot(X[(Y == -1).flatten(), 0], X[(Y == -1).flatten(), 1], 'rx')
plt.plot(np.array([-10, 10]), -(theta[2] + np.array([-10, 10])*theta[0])/theta[1], 'k-')
plt.show()
