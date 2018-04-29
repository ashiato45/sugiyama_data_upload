import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt("x")

v2 = np.loadtxt("out")

plt.axis([-20, 20, -20, 20])
lsp = np.linspace(-20, 20, 100)
plt.contourf(lsp, lsp, np.sign(v2))
plt.plot(x[0, :], x[1, :], 'ko')   
plt.plot(x[0, 0], x[1, 0], 'bo')   
plt.plot(x[0, 199], x[1, 199], 'ro')   
plt.show()