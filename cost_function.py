import dataset
import matplotlib.pyplot as plt
import numpy as np

xs,ys = dataset.get_beans(10)


plt.title("size-toxicity function",fontsize=12)
plt.xlabel("bean size")
plt.ylabel("toxicity")

#xsys is array [1,2,3,4]
plt.scatter(xs,ys)

w = 0.05

#predict function
y_pre = w * xs

print(y_pre)

plt.plot(xs,y_pre)

plt.show()

#均方差
es = (ys-y_pre)**2
sum_e = np.sum(es)/100

#plot函数相当于根据y=f(x)关系，先以x,y为坐标绘制结点，然后用直线连接结点。这里画了30个点
ws = np.arange(0,3,0.1)

#这里不是函数，而是循环算出每一个误差，打印出图像
es = []
for w in ws:
	y_pre = w*xs
	#会导致2条直线而底部是个∠，不平滑
	#e = (1/10)*np.sum(((ys-y_pre)**2)**0.5)
	e = (1/10)*np.sum((ys-y_pre)**2)
	es.append(e)


plt.title("cost function",fontsize=12)
plt.xlabel("w")
plt.ylabel("e")
plt.xlim([0,5])
plt.plot(ws,es)
plt.show()
