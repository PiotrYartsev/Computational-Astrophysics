from tqdm.notebook import tqdm
import math as math
import matplotlib.pyplot as plt
import scipy as scipy
import numpy as np


import jplephem as jpl 
from jplephem.spk import SPK
#get gravitational constant

#set the initial conditions

# pressure, velocity, eneergy, distance between particles, mass of particles, number of particles
initial_conditions_x_less_or_equal_0=[1,0,0,0,2.5,1,0.001875,0.001875,320]
initial_conditions_x_greater_0=[0.25,0,0,0,1.795,0.1795,0.0075,0.001875,80]


#Populate the x axis
x_less_than_o=np.linspace(0+0.0075,6,320)
x_greater_than_0=np.linspace(0,-6+0.001875,80)
x=np.concatenate((x_less_than_o,x_greater_than_0),axis=0)

#create an empty state vector
State_vector=np.zeros((len(x),len(initial_conditions_x_less_or_equal_0)+2))



#populate the state vector
for i in range(len(x)):
    if x[i]<=0:
        State_vector[i]=[x[i]]+[0]+[0]+initial_conditions_x_less_or_equal_0[:-1]
    else:
        State_vector[i]=[x[i]]+[0]+[0]+initial_conditions_x_greater_0[:-1]


#position in x direction 0
x=np.concatenate((x_less_than_o,x_greater_than_0),axis=0)


#define the smoothing length
d=1
h_1=1.3*(initial_conditions_x_less_or_equal_0[-2]/initial_conditions_x_less_or_equal_0[4])**(1/d)
h_2=1.3*(initial_conditions_x_greater_0[-2]/initial_conditions_x_greater_0[4])**(1/d)
h=(h_1+h_2)/2

#just use a defoult value
h_test=0.001875*5
h=h_test
a_d=1/h

#kernel functions
W=lambda R, a_d,h: a_d*(2/3-R**2+1/2*R**3) if R<=1 and R>=0 else a_d*(1/6*(2-R)**3) if R>1 and R<=2 else 0


#derivative of the kernel function
def W_derivat(R,a_d,h,dx):
    r=R*h
    if R<=1 and R>=0:
        return a_d*(-2+3/2*R)*dx/h**2
    elif R>1 and R<=2:
        return a_d*(-1/2(2-R)**2)*dx/(h*r)
    else:
        return 0

#position in x direction 0
#position in y direction 1
#position in z direction 2
#pressure 3 
#velocity in x direction 4
#velocity in y direction 5
#velocity in z direction 6
#energy 7
#mass 8
#number of particles 9


def G_function(State_vector,t=0):

    print(np.shape(State_vector))
    #create an empty derivative of the state vector
    State_vector_dir=np.zeros((len(x),len(initial_conditions_x_less_or_equal_0)+2))
    
    Delta_W=np.zeros((len(x),len(x)))
    Delta_W_dir=np.zeros((len(x),len(x)))

    new_pressure=np.zeros(len(x))
    new_velocity=np.zeros(len(x))
    new_energy=np.zeros(len(x))
    new_position=np.zeros(len(x))
    x_1=State_vector[:,0]
    sumxx=x_1-x_1[:,np.newaxis]
    for i in range(len(x)):
        dx=(sumxx[i])
        R=np.abs(dx)/h
        
        #define the derivative of pressure from the mass and velocity from the state vector



    #set the derivate of futere position as the speed
    State_vector_dir[:,0]=State_vector[:,4]
    State_vector_dir[:,1]=State_vector[:,5]
    State_vector_dir[:,2]=State_vector[:,6]
    
    return State_vector_dir
    

print(G_function(State_vector)[0])

"""
W=np.zeros((len(x),6))


for i in range(len(x)):
    W[i]=np.array([x[i],y[i],z[i],v_x[i],v_y[i],v_z[i]])


print(W)
energy_list = []

def Der_W(t,W):

    W_derivat=np.zeros(len(W))

    for i in range(int(len(W)/6)):
        W_derivat[i*6]=W[i*6+3]
        W_derivat[i*6+1]=W[i*6+4]
        W_derivat[i*6+2]=W[i*6+5]
        W_derivat[i*6+3]=0
        W_derivat[i*6+4]=0
        W_derivat[i*6+5]=0

    kinetic_energy = 0
    potential_energy = 0

    for i in range(int(len(W)/6)):
        kinetic_energy += 0.5*masses[i]*(W[i*6+3]**2+W[i*6+4]**2+W[i*6+5]**2)
        for j in range(int(len(W)/6)):
            if i!=j:
                r=math.sqrt((W[i*6]-W[j*6])**2+(W[i*6+1]-W[j*6+1])**2+(W[i*6+2]-W[j*6+2])**2)
                potential_energy += -G*masses[i]*masses[j]/r
                W_derivat[i*6+3]+=G*masses[j]*(W[j*6]-W[i*6])/r**3
                W_derivat[i*6+4]+=G*masses[j]*(W[j*6+1]-W[i*6+1])/r**3
                W_derivat[i*6+5]+=G*masses[j]*(W[j*6+2]-W[i*6+2])/r**3
    
    total_energy = kinetic_energy + potential_energy
    energy_list.append(total_energy)

    return W_derivat



#import a function to calculate the Runge-Kutta method
from scipy.integrate import RK45 as RK45

W=W.reshape(-1)


result = []

# Initialize the RK45 integrator
integrator = RK45(Der_W, t, W, t_end, h,atols=1e-20,rtols=1e-20)

# Integrate the equations of motion
pbar = tqdm(total=t_end) # set the total progress to the final time
while integrator.t < t_end:
    integrator.step()
    pbar.update(integrator.step_size) # update the progress bar by the step size
    final_state = integrator.y
    final_state=final_state.reshape((len(x),6))
    result.append(final_state)
pbar.close()

result=np.stack(result)

#for each state of the system, reshape the array to the original shape 

print(result)



#"""
