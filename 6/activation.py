import matplotlib.pyplot as plt
import dataset
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#从数据中获取随机豆豆
m=100
xs,ys = dataset.get_beans(m)


#配置图像
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel('Bean Size')
plt.ylabel('Toxicity')

# 豆豆毒性散点图
plt.scatter(xs, ys) 

#一变量第一神经元第一层
w11_1 = np.random.rand()
b1_1 = np.random.rand()
#一变量第二神经元第一层
w12_1 = np.random.rand()
b2_1 = np.random.rand()

#一变量第一神经元第二层
w11_2 = np.random.rand()
#二变量第一神经元第二层
w21_2 = np.random.rand()
b1_2 = np.random.rand()

def sigmoid(z):
    return 1/(1+np.exp(-z))

#前向传播,a1_2不是y，这里y用做样本去了，y-a1_2算e
#这东西其实是y=(x*@%%%！……@……)**#!&@的一大串的复合函数，但处理时是分一个个函数处理的，等于系统拆解成子系统，然后调子系统结构参数！
def forward_propgation(x):
    z1_1 = w11_1*x+b1_1
    a1_1 = sigmoid(z1_1)

    z2_1 = w12_1*x+b2_1
    a2_1 = sigmoid(z2_1)

    z1_2 = w11_2*a1_1+w21_2*a2_1 + b1_2
    a1_2 = sigmoid(z1_2)
    return a1_2,z1_2,a2_1,z2_1,a1_1,z1_1

a1_2,z1_2,a2_1,z2_1,a1_1,z1_1 = forward_propgation(xs)


#预测函数图像
plt.plot(xs,a1_2) 
#显示图像
plt.show()  



for j in range(5000):
    for i in range(100):
        #随机梯度下降，就是循环那100个样本 
        x = xs[i]
        y = ys[i]

        a1_2,z1_2,a2_1,z2_1,a1_1,z1_1 = forward_propgation(x)
        #error误差
        e = (y-a1_2)**2     
        #e = (y-a1_2)**2看成y = (a-x)**2；x1=a-x；dydx1=2x1；dx1dx=-1；所以等于2x1*-1=-2(a-x)
        deda1_2 = -2*(y-a1_2)

        #a2_1 = sigmoid(w12_1*xs+b2_1)
        da1_2dz1_2 = a1_2*(1-a1_2)

        dz1_2dw11_2 = a1_1
        dz1_2dw21_2 = a2_1

        dedw11_2 = deda1_2*da1_2dz1_2*dz1_2dw11_2
        dedw21_2 = deda1_2*da1_2dz1_2*dz1_2dw21_2

        dz1_2db1_2 = 1
        dedb1_2 = deda1_2*da1_2dz1_2*dz1_2db1_2

        dz1_2da1_1 = w11_2
        da1_1dz1_1 = a1_1*(1-a1_1)
        dz1_1dw11_1 = x
        dedw11_1 = deda1_2*da1_2dz1_2*dz1_2da1_1*da1_1dz1_1*dz1_1dw11_1
        dz1_1db1_1 = 1
        dedb1_1 = deda1_2*da1_2dz1_2*dz1_2da1_1*da1_1dz1_1*dz1_1db1_1

        dz1_2da2_1 = w21_2
        da2_1dz2_1 = a2_1*(1-a2_1)
        dz2_1dw12_1 = x
        dedw12_1 = deda1_2*da1_2dz1_2*dz1_2da2_1*da2_1dz2_1*dz2_1dw12_1
        dz2_1db2_1 = 1
        dedb2_1 = deda1_2*da1_2dz1_2*dz1_2da2_1*da2_1dz2_1*dz2_1db2_1
        

        #每一个模块的参数，进行调整，调整依据是这个参数对e的偏导数
        alpha = 0.03
        w11_2 = w11_2 - alpha*dedw11_2
        w21_2 = w21_2 - alpha*dedw21_2
        b1_2 = b1_2 - alpha*dedb1_2

        w12_1 = w12_1 - alpha*dedw12_1
        b2_1 = b2_1 - alpha*dedb2_1
        
        w11_1 = w11_1 - alpha*dedw11_1
        b1_1 = b1_1 - alpha*dedb1_1

    if(j%100==0):
        plt.clf() #clear window
        plt.scatter(xs,ys)
        a1_2,z1_2,a2_1,z2_1,a1_1,z1_1 = forward_propgation(xs)

        plt.plot(xs,a1_2)
        plt.pause(0.01)
    




plt.show()  