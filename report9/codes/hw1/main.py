import sys
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.io
import gc
import random

np.random.seed(42)

n = 50

# Prepare data
y = np.ones((n//2, 1))*np.array([1,-1])
y = y.reshape(y.size, 1).transpose()
y = np.ones((1, n))
y[0, n//2 + 1:] *= -1
# y = [1,..,1,-1,...,-1]
print(y.shape)
print(y)

x = np.random.randn(n, 2)
x[n//2 + 1:, 0] *= -1
x[0:n//2, 0] -= 15
x[n//2 + 1:, 0] -= 5
x[0:2, 0] += 10
x = np.hstack((x, np.ones((n, 1))))
x = x.transpose() # 3xn
print(x.shape)

mu = np.zeros((3, 1))
sigma = np.eye(3)
gamma = 0.5
indices = np.arange(n)
np.random.shuffle(indices)
for i in indices:
    beta = x[:, i].reshape(1, 3).dot(sigma).dot(x[:, i].reshape(3, 1)) + gamma
    mu += y[0, i]*np.maximum(0, 1-mu.transpose().dot(x[:, i])*y[0, i])/beta*sigma.dot(x[:, i].reshape(3, 1)).reshape(3, 1)
    print(sigma.dot(x[:, i]).reshape(3, 1).dot(x[:, i].reshape(1, 3)).shape)
    sigma -= sigma.dot(x[:, i]).reshape(3, 1).dot(x[:, i].reshape(1, 3)).dot(sigma)/beta

# print
plt.axis([-20, 0, -2.1, 2.1])
plt.plot(x[0, 0:n//2], x[1, 0:n//2], 'bo')   
plt.plot(x[0, n//2 + 1:], x[1, n//2 + 1:], 'rx')   
dots = np.linspace(-22, 2, 100)
plt.plot(dots, -(dots*mu[0] + mu[2])/mu[1], '-')
plt.show()