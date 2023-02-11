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

#Populate the x axis
x_less_than_o=np.linspace(0+0.0075,6,320)
x_greater_than_0=np.linspace(0,-6+0.001875,80)
x=np.concatenate((x_less_than_o,x_greater_than_0),axis=0)

#create an empty state vector
State_vector=np.zeros((len(x),len(initial_conditions_x_less_or_equal_0)+3))



#populate the state vector
for i in range(len(x)):
    if x[i]<=0:
        State_vector[i]=[x[i]]+[0]+[0]+initial_conditions_x_less_or_equal_0
    else:
        State_vector[i]=[x[i]]+[0]+[0]+initial_conditions_x_greater_0


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

    print(np.shape(State_vector))

    x=State_vector[:,0]
    y=State_vector[:,1]
    z=State_vector[:,2]
    density=State_vector[:,3]
    velocity_x=State_vector[:,4]
    velocity_y=State_vector[:,5]
    velocity_z=State_vector[:,6]
    energy=State_vector[:,7]



    #create an empty derivative of the state vector
    State_vector_dir=np.zeros((len(x),len(initial_conditions_x_less_or_equal_0)+3))

    der_x=State_vector_dir[:,0]
    der_y=State_vector_dir[:,1]
    der_z=State_vector_dir[:,2]
    der_density=State_vector_dir[:,3]
    der_velocity_x=State_vector_dir[:,4]
    der_velocity_y=State_vector_dir[:,5]
    der_velocity_z=State_vector_dir[:,6]
    der_energy=State_vector_dir[:,7]
    

    #define the x vector and calcualte the W_ij and delta W_ij

    #r-vector with sign
    r_sign=x-x[:,np.newaxis]

    
    
    
    #r-vector without sign
    r=np.sqrt(r_sign**2)
    #print(r)

    R=r/h
    #print(R)

    W_value=np.zeros((len(x),len(x)))

    Delta_W_value=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            R[i,j]=float(R[i,j])
            r[i,j]=float(r[i,j])
            r_sign[i,j]=float(r_sign[i,j])
            
            W_value[i,j]=W(R[i,j],r[i,j],a_d,h)
            Delta_W_value[i,j]=W_derivat(R[i,j],r[i,j],a_d,h,r_sign[i,j])
    
    #set the derivative of density to be 0 and set current density to be a fucntion
    State_vector_dir[:,3]=0
    for i in range(len(x)):
        State_vector[i,3]=sum(W_value[i,:]*mass_of_particle)

    #set the derivate of futere position as the speed
    der_x=velocity_x
    der_y=velocity_y
    der_z=velocity_z

    #set the derivative of the energy
    #c=math.sqrt((1.4-1)*State_vector[:,7])
    gamma=1.4
    pressure=np.zeros(len(x))
    pressure=(gamma-1)*density*energy
    seed_of_sound=np.zeros(len(x))
    seed_of_sound=np.sqrt((gamma-1)*energy)


    State_vector_dir=np.zeros((len(x),len(initial_conditions_x_less_or_equal_0)+3))

    State_vector_dir[:,0]=der_x
    State_vector_dir[:,1]=der_y
    State_vector_dir[:,2]=der_z
    State_vector_dir[:,3]=der_density
    State_vector_dir[:,4]=der_velocity_x
    State_vector_dir[:,5]=der_velocity_y
    State_vector_dir[:,6]=der_velocity_z
    State_vector_dir[:,7]=der_energy


    return State_vector_dir
    
print(State_vector[0])
print(G_function(State_vector)[0])
print(State_vector[0])
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

