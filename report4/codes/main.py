import sys
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

datanum = 10
datasamples = np.linspace(-3, 3, datanum)
datasamples = datasamples.reshape(len(datasamples), 1)
y = datasamples + 0.2*np.random.randn(datanum, 1)
y[datanum - 1] = -4
y[datanum - 2] = -4
y[3] = -4

eta = 1
theta = np.zeros((2, 1))
phi = np.ones((datanum, 2))
phi[:, 1] = datasamples.reshape(len(datasamples))
for i in range(10000):
    r = np.abs(np.dot(phi, theta) - y)
    w = np.zeros((datanum, 1))
    w[r <= eta] = ((1-r**2/eta**2)**2)[r <= eta]
    W = np.diag(w.reshape(len(w)))
    A = np.dot(np.dot(phi.transpose(), W), phi)
    b = np.dot(np.dot(phi.transpose(), W), y)
    theta2 = np.linalg.solve(A, b)
    if np.linalg.norm(theta2 - theta) < 0.001:
        break
    theta = theta2

graphnum = 5000
graphsamples = np.linspace(-3, 3, graphnum)
graphsamples = graphsamples.reshape(graphnum, 1)
graph = theta[0, 0] + theta[1, 0]*graphsamples

plt.axis([-3.2, 3.2, -4.2, 4.2])    
plt.plot(datasamples, y, 'o')
plt.plot(graphsamples, graph, '-')
plt.show()
