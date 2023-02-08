import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math as math

def W(r, h):
    if r>0:
        if r <= h:
            return -2/3 * (1 - r/h)**3+1/6 * (2 - r/h)**3
        elif r <= h*2 and r > h:
            return 1/6 * (2 - r/h)**3
        else:
            return 0
    else:
        if r >= -h:
            return -2/3 * (1 + r/h)**3+1/6 * (2 + r/h)**3
        elif r >= -h*2 and r < -h:
            return 1/6 * (2 + r/h)**3
        else:
            return 0

def W_derivat(r,h):
    if r>0:
        if r <= h:
            return 2*(1 - r/h)**2- 3/6*(2 - r/h)**2
        elif r <= h*2 and r > h:
            return -3/6 * (2 - r/h)**2
        else:
            return 0
    else:
        if r >= -h:
            return -2*(1 + r/h)**2+ 3/6*(2 + r/h)**2
        elif r >= -h*2 and r < -h:
            return 3/6 * (2 + r/h)**2
        else:
            return 0


h = 1
#a_d=15/(7*h**2*math.pi)
a_d=1
R = np.linspace(-2*h, 2*h, 100)
W_values = [a_d*W(r, h) for r in R]
W_values2=[]
for i in range(len(W_values)):
    for ii in range(len(W_values)):
        W_values2.append([R[i],R[ii],W_values[i]*W_values[ii]])

W_values2=np.array(W_values2)
W_values2=W_values2.reshape(len(R),len(R),3)

W_derivat_values = [a_d*W_derivat(r, h) for r in R]
W_derivat_values2=[]
for i in range(len(W_derivat_values)):
    for ii in range(len(W_derivat_values)):
        W_derivat_values2.append([R[i],R[ii],W_derivat_values[i]*W_derivat_values[ii]])

W_derivat_values2=np.array(W_derivat_values2)
W_derivat_values2=W_derivat_values2.reshape(len(R),len(R),3)

#make the result for -2h<x<2h and -2h<y<2h by using the symmetry



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#set opacity of the surface to 0.5
ax.plot_surface(W_values2[:,:,0], W_values2[:,:,1], W_values2[:,:,2],color='r',label='W(r)')



ax.set_xlabel('R(x)')
ax.set_ylabel('R(y)')
#write title in latex style
ax.set_title('2-D: '+ r'$W(R,h)\cdot a_d$')


ax.set_zlabel(r'$W(R,h)\cdot a_d$')
#add title

#plt.show()
plt.savefig('W(r,h).png')
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#set opacity of the surface to 0.5
ax.plot_surface(W_derivat_values2[:,:,0], W_derivat_values2[:,:,1], W_derivat_values2[:,:,2],color='r',label='W(r)')
ax.set_xlabel('R(x)')
ax.set_ylabel('R(y)')
ax.set_zlabel(r'$\frac{dW(R,h)}{dr}\cdot a_d$')
#add title
plt.title('2-D: '+ r'$\frac{dW(R,h)}{dr}\cdot a_d$')
#plt.show()
plt.savefig('W_dir(r,h).png')
plt.close()

