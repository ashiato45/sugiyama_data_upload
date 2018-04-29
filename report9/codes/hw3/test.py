import numpy as np
import matplotlib.pyplot as plt

theta = np.loadtxt("theta")
theta2 = np.loadtxt("theta2")

tx = np.loadtxt("tx")
px = np.loadtxt("px")

plt.axis([-5, 5, -10, 10])
p = np.linspace(-5, 5, 10)
plt.plot(p, -(theta2[2] + p*theta2[0])/theta2[1], 'g-')   
plt.plot(px[0, 0:90], px[1, 0:90], 'bo')   
plt.plot(px[0, 90:], px[1, 90:], 'ro')   
plt.show()

plt.axis([-5, 5, -10, 10])
p = np.linspace(-5, 5, 10)
plt.plot(p, -(theta[2] + p*theta[0])/theta[1], 'g-')   
plt.plot(tx[0, 0:10], tx[1, 0:10], 'bo')   
plt.plot(tx[0, 10:], tx[1, 10:], 'ro')   
plt.show()