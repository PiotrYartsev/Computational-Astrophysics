from tqdm.notebook import tqdm
import math as math
import matplotlib.pyplot as plt
import scipy as scipy
import numpy as np



#set the initial conditions

# density, velocity, pressure, eneergy, distance between particles
initial_conditions_x_less_or_equal_0=[1,0,0,0,2.5]
initial_conditions_x_greater_0=[0.25,0,0,0,1.795]

mass_of_particle=0.001875
visc=0

#Populate the x axis
x_less_than_o=np.linspace(0+0.0075,6,320)
x_greater_than_0=np.linspace(0,-6+0.001875,80)
x_list=np.concatenate((x_less_than_o,x_greater_than_0),axis=0)

#create an empty state vector
State_vector=np.zeros((len(x_list),len(initial_conditions_x_less_or_equal_0)+3))



#populate the state vector
for i in range(len(x_list)):
    if x_list[i]<=0:
        State_vector[i]=[x_list[i]]+[0]+[0]+initial_conditions_x_less_or_equal_0
    else:
        State_vector[i]=[x_list[i]]+[0]+[0]+initial_conditions_x_greater_0


#position in x direction 0
x=np.concatenate((x_less_than_o,x_greater_than_0),axis=0)


#define the smoothing length
d=1
h_1=1.3*(initial_conditions_x_less_or_equal_0[-2]/initial_conditions_x_less_or_equal_0[4])**(1/d)
h_2=1.3*(initial_conditions_x_greater_0[-2]/initial_conditions_x_greater_0[4])**(1/d)
h=(h_1+h_2)/2

#just use a defoult value
h_test=0.001875*20
h=h_test
a_d=1/h



#kernel functions
def W(R,r,a_d,h):
    if R<=1 and R>=0 or R==0:
        output=a_d*(2/3-R**2+1/2*R**3)
    elif R>1 and R<=2:
        output=a_d*(1/6*(2-R)**3)
    else:
        output=0
    return output




#derivative of the kernel function
def W_derivat(R,r,a_d,h,dx):
    if R<=1 and R>=0 or R==0:
        output=a_d*(-2+3/2*R)*dx/h**2
    elif R>1 and R<=2:
        output=a_d*(-(1/2)*(2-R)**2)*dx/(h*r)
    else:
        output=0
    return output

def density_function(mass,velocity_i, velocity_j, delta_W_ij):
    return mass*(velocity_i-velocity_j)*delta_W_ij

def velocity_function(mass,velocity_i, velocity_j,pressure_i,pressure_j,density_i, density_j, artvisc, delta_W_ij):
    return -mass*(pressure_i/(density_i**2)+pressure_j/(density_j**2)+artvisc)*delta_W_ij

def energy_function(mass,velocity_i, velocity_j,pressure_i,pressure_j,density_i, density_j, artvisc, delta_W_ij):
    return -1/2 * mass*(velocity_i*pressure_i/(density_i**2)+velocity_j*pressure_j/(density_j**2)+artvisc*velocity_i)*delta_W_ij
#position in x direction 0
#position in y direction 1
#position in z direction 2
#density 3 
#velocity in x direction 4
#velocity in y direction 5
#velocity in z direction 6
#energy 7


def G_function(State_vector,t=0):
    State_vector_derivat=np.zeros(len(State_vector))
    for i in range(8):
        State_vector_derivat[i*8]=State_vector[i*8+4]

