import dataset
import plot_utils

import numpy as np


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

m = 100
X,Y = dataset.get_beans4(m)
plot_utils.show_scatter(X, Y)

# 按照sequential的结构，组建神经网络的组分
model = Sequential()
model.add(Dense(units=2,activation='sigmoid',input_dim=2))
model.add(Dense(units=1,activation='sigmoid'))

# 配置神经网络的训练规则
model.compile(loss='mean_squared_error',optimizer=SGD(lr=0.1),metrics=['accuracy'])

# 开始训练神经网络
model.fit(X,Y,epochs=5000,batch_size=10)
pres = model.predict(X)

plot_utils.show_scatter_surface(X,Y,model)

