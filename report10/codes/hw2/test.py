import numpy as np
import matplotlib.pyplot as plt

wvec = np.loadtxt("worstVec")

matx = np.loadtxt("points")

plt.axis([-5, 5, -5, 5])
lsp = np.linspace(-20, 20, 100)
plt.plot(matx[0, :], matx[1, :], 'rx')     
plt.plot(lsp, lsp*(wvec[1]/wvec[0]), '-')  
plt.show()