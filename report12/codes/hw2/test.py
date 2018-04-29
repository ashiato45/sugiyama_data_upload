import numpy as np
import matplotlib.pyplot as plt

wvec = np.loadtxt("bVec")

matx = np.loadtxt("matX")

plt.axis([-5, 5, -5, 5])
lsp = np.linspace(-20, 20, 100)
plt.plot(matx[0, :50], matx[1, :50], 'rx')    
plt.plot(matx[0, 50:], matx[1, 50:], 'bx')    
plt.plot(lsp, lsp*(wvec[1]/wvec[0]), 'k-')  
plt.show()