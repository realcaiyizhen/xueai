import dataset
import matplotlib.pyplot as plt
import numpy as np

xs,ys = dataset.get_beans(100)


plt.title("size-toxicity function",fontsize=12)
plt.xlabel("bean size")
plt.ylabel("toxicity")
plt.xlim(0,1)
plt.ylim(0,1.5)

#xsys is array [1,2,3,4]
plt.scatter(xs,ys)
plt.show()