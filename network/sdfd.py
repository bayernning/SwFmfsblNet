import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,9,10)
plt.plot(x)
plt.savefig("test_plot.png")
plt.close()