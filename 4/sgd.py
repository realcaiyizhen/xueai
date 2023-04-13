import dataset
import matplotlib.pyplot as plt
import numpy as np

xs,ys = dataset.get_beans(100)


w = 0.1
b = 0.1


#这里是每个点，只运行一次梯度下降，运行过后就拿下一个点来做对比了

for j in range(30):
    for i in range(100):
        # 这里越靠近100的点，下降的越快，越远离翘点靠近右边
        
        x = xs[i]
        y = ys[i]
        

        dw = 2*x**2*w + 2*x*b - 2*x*y
        db = 2*b + 2*x*w - 2*y
        print (dw)

        alpha = 0.3
        w = w - alpha*dw
        b = b - alpha*db

    plt.clf() #clear window
    plt.scatter(xs,ys)
    y_pre = w*xs+b
    plt.plot(xs,y_pre)
    plt.pause(0.01)
    
