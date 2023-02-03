import numpy as np

x = np.linspace(7, 42, 100)
y = np.linspace(96, 0, 100)

print(x)
print(x.shape)
print(y)

sumxy=x+y[:,np.newaxis]

print(sumxy)
#get the shape of the array
print(sumxy.shape)