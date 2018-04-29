import sys
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

h = float(sys.argv[1])
lam = float(sys.argv[2])

datanum = 50
datasamples = np.linspace(-3, 3, datanum)
datasamples = datasamples.reshape(len(datasamples), 1)
y = np.sin(np.pi*datasamples)/(np.pi*datasamples) + 0.1*datasamples
y += np.random.randn(datanum, 1)*0.2 # normal distribution
# print(y)

def gk(x, c):
    return np.exp(-(x-c)**2/(2*(h**2)))

phi = np.fromfunction(lambda i, j: gk(datasamples[i, 0], datasamples[j, 0]), (datanum, datanum), dtype=int)

theta = np.zeros((datanum, 1))
z = np.zeros((datanum, 1))
u = np.zeros((datanum, 1))

while True:
    theta2 = np.linalg.solve(np.dot(phi.transpose(), phi) + np.eye(datanum, dtype=int), np.dot(phi.transpose(), y) + z - u)
    z2 = np.maximum(0, theta2+u-lam) + np.minimum(0, theta2+u+lam)
    u2 = u + theta2 - z2
    dt = theta-theta2
    dz = z-z2
    du = u-u2
    theta = theta2
    z = z2
    u = u2
    if np.all(abs(dt) < 1e-9) and np.all(abs(dz) < 1e-9) and np.all(abs(du) < 1e-9):
        break
        

graphnum = 5000
graphsamples = np.linspace(-3, 3, graphnum)
graphsamples = graphsamples.reshape(graphnum, 1)
calckernel = lambda x: np.dot(theta.transpose(), np.fromfunction(lambda i, j:gk(x, datasamples[i, 0]), (datanum, 1), dtype=int))[0, 0]
km = np.fromfunction(lambda i,j: gk(graphsamples[i, 0], datasamples[j, 0]), (graphnum, datanum), dtype=int)
graph = np.dot(km, theta)

em = np.fromfunction(lambda i,j: gk(datasamples[i, 0], datasamples[j, 0]), (datanum, datanum), dtype=int)
eout = np.dot(em, theta)
er = eout - y

print(theta)
print(np.sum(abs(theta) < 1e-8))
print(np.linalg.norm(er))

plt.axis([-2.8, 2.8, -1, 1.5])    
plt.plot(datasamples, y, 'o')
plt.plot(graphsamples, graph, '-')
# plt.plot(datasamples, eout, 'o')
plt.show()
