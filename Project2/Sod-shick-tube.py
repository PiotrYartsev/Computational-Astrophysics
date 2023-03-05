from tqdm.notebook import tqdm
import math as math
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm 
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
#set the initial conditions

# density, velocity x, velocity y, velocity z, energy
initial_conditions_x_less_or_equal_0=[1,0,0,0,2.5]
initial_conditions_x_greater_0=[0.25,0,0,0,1.795]

mass_of_particle=0.001875

#Populate the x axis
small_step=0.6/80
start=-0.6
State_vector=[]
for i in np.linspace(-0.6,0,320):
    State_vector.append((i, 0, 0, initial_conditions_x_less_or_equal_0[0], initial_conditions_x_less_or_equal_0[1], initial_conditions_x_less_or_equal_0[2], initial_conditions_x_less_or_equal_0[3], initial_conditions_x_less_or_equal_0[4]))
for i in np.linspace(0+small_step,0.6,80):
    State_vector.append((i, 0, 0, initial_conditions_x_greater_0[0], initial_conditions_x_greater_0[1], initial_conditions_x_greater_0[2], initial_conditions_x_greater_0[3], initial_conditions_x_greater_0[4]))
State_vector=np.array(State_vector)
x=State_vector[:,0]
number_of_particles=len(x)


#d=1
#h_1=1.3*(mass_of_particle/initial_conditions_x_less_or_equal_0[0])**(1/d)
#h_2=1.3*(mass_of_particle/initial_conditions_x_greater_0[0])**(1/d)


#kernel functions
def W(dx,h):
    a_d=1/h
    r=norm(dx)
    R=r/h
    if R<=1 and R>=0:
        return a_d * (2/3 - R**2 + 1/2 * R**3)
    else:
        return a_d * (1/6 * (2-R)**3)


#derivative of the kernel function
def W_derivat(dx,h):
    a_d=1/h
    #print("a_d",a_d)
    r=norm(dx)
    #print("r",r)
    R=r/h
    #print("R",R)
    if R<=1 and R>=0:
        return a_d * (-2 + 3/2 * R) * dx / h**2
    else:
        return -a_d * ((1/2) * (2 - R)**2) * dx / (h * r)
    


#position in x direction 0
#position in y direction 1
#position in z direction 2
#density 3 
#velocity in x direction 4
#velocity in y direction 5
#velocity in z direction 6
#energy 7
import time as time

def G_function(State_vector,t):
    State_vector_dir = np.zeros((number_of_particles, len(initial_conditions_x_less_or_equal_0) + 3))
    x_values=State_vector[:,0]
    y_values=State_vector[:,1]
    z_values=State_vector[:,2]
    density=State_vector[:,3]
    velocity_x=State_vector[:,4]
    velocity_y=State_vector[:,5]
    velocity_z=State_vector[:,6]
    energy=State_vector[:,7]


    Derivative_x_values=State_vector_dir[:,0]
    Derivative_y_values=State_vector_dir[:,1]
    Derivative_z_values=State_vector_dir[:,2]
    Derivative_density=State_vector_dir[:,3]
    Derivative_velocity_x=State_vector_dir[:,4]
    Derivative_velocity_y=State_vector_dir[:,5]
    Derivative_velocity_z=State_vector_dir[:,6]
    Derivative_energy=State_vector_dir[:,7]

    #h-list
    #defoult values for right and left side
    
    h_1=0.002*2.5
    h_2=h_1*5
    
    """
    h_1=0.006
    h_2=0.006
    """
    #create a list of h values
    h_list=[]

    #add the default values to the list
    for i in range(320):
        h_list.append(h_1)
    for i in range(320,number_of_particles):
        h_list.append(h_2)

    #calculate the distance between the particles
    dx = x_values - x_values[:, np.newaxis]

    #define the kernel function
    W_value = np.zeros((number_of_particles, number_of_particles))

    #define the derivative of the kernel function
    Delta_W_value = np.zeros((number_of_particles, number_of_particles))

    #calculate the kernel function and the derivative of the kernel function
    k=2
    for i in range(number_of_particles):
        for j in range(number_of_particles):
            h_ij=(h_list[i]+h_list[j])/2
            if norm(dx[i, j])<=(k*(h_ij)):
                W_value[i, j] = W(dx[i,j], h_ij)
                Delta_W_value[i, j] = W_derivat(dx[i, j], h_ij)

    
    for i in range(number_of_particles):
        density[i]=mass_of_particle*np.sum(W_value[i,:])
        Derivative_density = 0


    gamma = 1.4
    pressure = (gamma - 1) * density * energy

    visc=0

    #calculte the derivative of poition as the velocity
    Derivative_x_values = velocity_x
    Derivative_y_values = velocity_y
    Derivative_z_values = velocity_z

    #density
    for i in range(number_of_particles):
        #define the derivative of density, velocity and energy starting with         

        der_velocity_i=0
        der_energy_i=0

        #calculate the sums used for the derivative of velocity and energy
        for j in range(number_of_particles):
            #calculate the pressure for the particle j if condition is true
            if Delta_W_value[i,j]!=0:

                #calculate the derivative of velocity
                der_velocity_i+= (-1)*mass_of_particle*(pressure[i]/density[i]**2 + pressure[j]/density[i]**2+visc)*Delta_W_value[i,j]

                #calculate the derivative of energy
                der_energy_i+= 1/2 * mass_of_particle * (pressure[i] /density[i]**2 + pressure[j]/density[i]**2+visc) * (velocity_x[i] -velocity_x[i]) * Delta_W_value[i,j]

        if der_energy_i<0:
            print("negative energy",der_energy_i)
            der_energy_i=0
        
        #add the  velocity and energy to the derivative state vector
        Derivative_velocity_x[i] = der_velocity_i
        Derivative_energy[i] = der_energy_i
    
    #set the values to 0 for particles within 10 of the walls
    """
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].plot(x_values, density, 'o')
    axs[0].set_title('density')
    axs[1].plot(x_values, pressure, 'o')
    axs[1].set_title('pressure')
    axs[2].plot(x_values, Derivative_velocity_x, 'o')
    axs[2].set_title('der_velocity')
    axs[3].plot(x_values, Derivative_energy, 'o')
    axs[3].set_title('der_energy')
    plt.show()
    """

    State_vector_dir[:,0]=Derivative_x_values
    State_vector_dir[:,1]=Derivative_y_values
    State_vector_dir[:,2]=Derivative_z_values
    State_vector_dir[:,3]=Derivative_density
    State_vector_dir[:,4]=Derivative_velocity_x
    State_vector_dir[:,5]=Derivative_velocity_y
    State_vector_dir[:,6]=Derivative_velocity_z
    State_vector_dir[:,7]=Derivative_energy

    State_vector[:,0]=x_values
    State_vector[:,1]=y_values
    State_vector[:,2]=z_values
    State_vector[:,3]=density
    State_vector[:,4]=velocity_x
    State_vector[:,5]=velocity_y
    State_vector[:,6]=velocity_z
    State_vector[:,7]=energy


    return State_vector_dir,W_value,Delta_W_value

State_vector_dir,W_value,Delta_W_value=G_function(State_vector, 0)

    