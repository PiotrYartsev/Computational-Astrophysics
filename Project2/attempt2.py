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
    r=np.where(np.abs(r)<=k*h_vect,r,0)
    #create a W matrix
    W_matrix=np.zeros((len(State_vector),len(State_vector)))
    #populate the W matrix using broadcasting
    W_matrix=np.where(r!=0,W(r,h_vect),0)
    W_deriv_matrix=np.zeros((len(State_vector),len(State_vector),3))
    W_deriv_matrix=np.where(r!=0,W_gradient(r,h_vect),0)


    
    #calculate the velocity
    pressure=[(gamma-1)*(State_vector[i][-1])*(State_vector[i][-2]) for i in range(len(State_vector))]
    #make pressure2 a 400x400 matrix where the elements are pressure_i/density_i**2 + pressure_j/density_j**2
    pressure2=np.zeros((len(State_vector),len(State_vector)))
    for i in range(len(State_vector)):
        pressure2[i]=pressure[i]/State_vector[i][-2]**2
    pressure2=pressure2+pressure2.T
    
    
    State_vector_dir[:,3]=-mass_of_particle*np.sum(W_deriv_matrix*(pressure2+gamma),axis=1)
    
    #calculate the energy
    #make a velocity matrix 400x400 v_i-v_j fot v_x
    velocity_matrix=np.zeros((len(State_vector),len(State_vector)))
    for i in range(len(State_vector)):
        velocity_matrix[i]=State_vector[i][3]-State_vector[:,3]

    
    State_vector_dir[:,7]=(1/2)*mass_of_particle*np.sum(W_deriv_matrix*(pressure2+gamma)*velocity_matrix,axis=1)
    #set the derivative of the position to the velocity
    State_vector_dir[:,0]=State_vector[:,3]
    State_vector_dir[:,1]=State_vector[:,4]
    State_vector_dir[:,2]=State_vector[:,5]
    
    #calculate the density equal to zero
    State_vector[:,6]=mass_of_particle*np.sum(W_matrix,axis=1)
    State_vector_dir[:,6]=0

    return State_vector_dir


#print(SPH(State_vector)[0][320],SPH(State_vector)[1][320])

def RK4(State_vector):
    
    k1=SPH(State_vector)
    k2=SPH(State_vector+0.5*k1)
    k3=SPH(State_vector+0.5*k2)
    k4=SPH(State_vector+k3)
    State_vector=State_vector+(1/6)*(k1+2*k2+2*k3+k4)
    return State_vector

t_0=0
h=0.005
t_f=h*40

t_list=np.arange(t_0,t_f,h)
result=np.zeros((len(t_list)+1,len(State_vector),len(State_vector[0])))
print(len(t_list))
print(result.shape)
result[0]=State_vector
for i in range(len(t_list)):
    State_vector=RK4(State_vector)
    print(i)
    result[i+1]=State_vector

for section in result[10:12]:
    x=section[:,0]
    print(len(x))
    velocity_x=section[:,3]
    density=section[:,6]
    pressure=[(gamma-1)*(section[i][-1])*(section[i][-2]) for i in range(len(section))]
    energy=section[:,7]


    #order all list by x
    x, density, pressure, velocity_x, energy = zip(*sorted(zip(x, density, pressure, velocity_x, energy)))
    #make 4 plots side by side
    fig, axs = plt.subplots(1, 4,figsize=(20,5))
    axs[0].plot(x,density)
    axs[0].set_title('density')
    axs[1].plot(x,pressure)
    axs[1].set_title('pressure')
    axs[2].plot(x,velocity_x)
    axs[2].set_title('velocity_x')
    axs[3].plot(x,energy)
    axs[3].set_title('energy')
    plt.show()