from tqdm.notebook import tqdm
import math as math
import matplotlib.pyplot as plt
import scipy as scipy
import numpy as np
from numpy.linalg import norm 
import sys
import numpy

x=[]
for i in range(320):
    x.append(0.001875*i-0.6)

for i in range(80):
    x.append(0.0075*i)



mass_of_particle=0.001875
#starting state vector: velocity (x,y,z), density, energy
Starting_values_x_less0=[0,0,0,1,2.5]
Starting_values_x_greater0=[0,0,0,0.25,1.795]

State_vector=np.zeros((len(x),len(Starting_values_x_less0)+3))
#final state vector:position (x,y,z), velocity (x,y,z), density, energy
for i in range(len(x)):
    if x[i]<=0:
        temp=[x[i],0,0]+Starting_values_x_less0
        State_vector[i]=temp
    else:
        temp=[x[i],0,0]+Starting_values_x_greater0
        State_vector[i]=temp
        

def W(r_ij,h_average_ij):
    R = np.linalg.norm(r_ij,axis=1)/h_average_ij
    a_d = 1/h_average_ij
    
    W1 = a_d*(2/3-R**2 - 1/2*R**3)
    W2 = a_d*(1/6*(2-R)**3)
    
    return np.where((R>=0) & (R<=1), W1, W2)
    

def W_gradient(r_ij,h_average_ij):
    R = np.linalg.norm(r_ij,axis=1)/h_average_ij
    a_d = 1/h_average_ij
    r = np.linalg.norm(r_ij,axis=1)
    dx = r_ij
    
    W1 = a_d*(-2+3/2*R)*(dx/h_average_ij**2)
    W2 = -a_d*1/2*(2-R)**2  * (dx/(h_average_ij*r))
    
    return np.where((R>=0) & (R<=1), W1, W2)

#constants
gamma=1.4
visc=0

def SPH(State_vector):

    State_vector_dir=np.zeros((len(State_vector),len(State_vector[0])))

    pressure=[(gamma-1)*(State_vector[i][-1])*(State_vector[i][-2]) for i in range(len(State_vector))]
    
    #set the derivative of the position to the velocity
    State_vector_dir[:,0]=State_vector[:,3]
    State_vector_dir[:,1]=State_vector[:,4]
    State_vector_dir[:,2]=State_vector[:,5]
    #print(State_vector_dir)
    
    h_1=0.01878527*5
    h_2=0.01878527

    
    
    h_list=np.zeros(len(State_vector))
    for i in range(len(State_vector)):
        if State_vector[i][0]<=0:
            h_list[i]=h_1
        else:
            h_list[i]=h_2
    h_vect=(h_list+h_list.reshape(-1,1))/2
    #calculate W for all particles
    r=State_vector[:,0]-State_vector[:,0].reshape(-1,1)
    #if a value of r is less than 1/h_vect in the same position, set it to 0

    k=2
    r=np.where(abs(r)<=k*h_vect,r,0 )
    print(r.shape)
    print(h_vect.shape)
    
    #create a W matrix
    W_matrix=np.zeros((len(State_vector),len(State_vector)))
    #populate the W matrix using broadcasting
    W_matrix=W(r,h_vect)
    #set the diagonal to 0
    np.fill_diagonal(W_matrix,0)
    
    
    


SPH(State_vector)