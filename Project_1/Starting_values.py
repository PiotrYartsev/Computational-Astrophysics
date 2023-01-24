import tqdm as tqdm
import math as math

#get gravitational constant
G=4*math.pi**2

h=0.01
t=0
t_end=300

Sun_mass=1
Jupiter_mass=0.001
Trojan1_mass=0
Trojan2_mass=0

x=[0,0,-4.503,4.503]
y=[0, 5.2, 2.6, 2.6]
z=[0,0,0,0]

v_x=[0, -2.75674, -1.38, -1.38]
v_y=[0,0, -2.39, 2.39]
v_z=[0,0,0,0]

#make an matrix of the starting values
import numpy as np

W=np.array([[x[0],y[0],z[0],v_x[0],v_y[0],v_z[0]],[x[1],y[1],z[1],v_x[1],v_y[1],v_z[1]],[x[2],y[2],z[2],v_x[2],v_y[2],v_z[2]],[x[3],y[3],z[3],v_x[3],v_y[3],v_z[3]]])


print(W)


print(len(W))
print(len(W[0]))






def Der_W(W,t):

    #make a 4x6 matrix of zeros
    W_derivat=np.zeros((len(W),len(W[0])))
    for i in range(3):
        W_derivat[:,i]=W[:,i]+W[:,i+3]

    for ii in [0,1,2]:
        r=W[:,ii]
        len_vector_r=np.sqrt(r[0]**2+r[1]**2+r[2]**2)

        print(r)
        print(len_vector_r)


    return(W_derivat)
    

    

print(Der_W(W,t))



#"""


