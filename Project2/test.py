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
visc=0

#Populate the x axis
small_step=0.6/80
start=-0.6
State_vector=[]
for i in np.linspace(-0.6,0,320):
    State_vector.append((i, 0, 0, initial_conditions_x_less_or_equal_0[0], initial_conditions_x_less_or_equal_0[1], initial_conditions_x_less_or_equal_0[2], initial_conditions_x_less_or_equal_0[3], initial_conditions_x_less_or_equal_0[4]))
for i in np.linspace(0,0.6,80):
    State_vector.append((i, 0, 0, initial_conditions_x_greater_0[0], initial_conditions_x_greater_0[1], initial_conditions_x_greater_0[2], initial_conditions_x_greater_0[3], initial_conditions_x_greater_0[4]))
State_vector=np.array(State_vector)
x=State_vector[:,0]
number_of_particles=len(x)
#print(x)
print(State_vector.shape)
print(State_vector[0])

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
    
def vesc_func(c_ij,density_ij,phi_ij):
    alpha=1
    beta=1
    return (-alpha*c_ij*phi_ij+beta*phi_ij*phi_ij)/(density_ij)

def phi_func(h_ij,v_ij,x_ij):
    varphi=0.1*h_ij
    return (h_ij*v_ij*x_ij)/(abs(x_ij)**2+varphi**2)


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

    #h-list
    #defoult values for right and left side
    h_1=0.002*2.5
    h_2=h_1*5
    
    #create a list of h values
    h_list=[]

    #add the default values to the list
    for i in range(320):
        h_list.append(h_1)
    for i in range(320,number_of_particles):
        h_list.append(h_2)


    k=2
    #calculate the distance between the particles
    dx = State_vector[:, 0] - State_vector[:, 0][:, np.newaxis]

    #calculate the velocity between the particles
    dv=State_vector[:,4]-State_vector[:,4][:,np.newaxis]

    #define the kernel function
    W_value = np.zeros((number_of_particles, number_of_particles))

    #define the derivative of the kernel function
    Delta_W_value = np.zeros((number_of_particles, number_of_particles))

    #define the viscosity function
    #visc=np.zeros((number_of_particles,number_of_particles))

    #define the pressure
    gamma = 1.4
    pressure = (gamma - 1) * State_vector[:, 3] * State_vector[:, 7]

    #define the seed of sound 
    seed_of_sound = np.sqrt((gamma - 1) * State_vector[:, 7])

    #calculate the kernel function and the derivative of the kernel function
    for i in range(number_of_particles):
        for j in range(number_of_particles):
            h_ij=(h_list[i]+h_list[j])/2
            c_ij=(seed_of_sound[i]+seed_of_sound[j])/2
            if i!=j:
                if norm(dx[i, j])<=(k*(h_ij)):
                    W_value[i, j] = W(dx[i,j], h_ij)
                    Delta_W_value[i, j] = W_derivat(dx[i, j], h_ij)
                """if dx[i,j]*dv[i,j]<0:
                    phi=phi_func(h_ij,dv[i,j],dx[i,j])
                    visc[i,j]=vesc_func(c_ij,State_vector[i,3],phi)
                #"""

    visc=0

    #calculte the derivative of poition as the velocity
    State_vector_dir[:, 0] = State_vector[:, 4]
    State_vector_dir[:, 1] = State_vector[:, 5]
    State_vector_dir[:, 2] = State_vector[:, 6]

    
    
    for i in range(number_of_particles):
        #calculate the pressure for the particle i
        pressure_i = pressure[i] 
        #same with dentsity
        density_i = State_vector[i, 3] 
        #same for velocity
        velocity_i = State_vector[i, 4] 
        
        #define the derivative of density, velocity and energy starting with 0
        der_density_i=0
        der_velocity_i=0
        der_energy_i=0

        #calculate the sums used for the derivative of density, velocity and energy
        for j in range(number_of_particles):
            #calculate the pressure for the particle j if condition is true
            if i!=j and Delta_W_value[i,j]!=0:

                #calculate the derivative of density
                der_density_i+=mass_of_particle*(velocity_i-State_vector[j,4])*W_value[i,j]

                #calculate the derivative of velocity
                der_velocity_i+=-mass_of_particle*(pressure_i/density_i**2 + pressure[j]/State_vector[j,3]**2+visc)*Delta_W_value[i,j]

                #calculate the derivative of energy
                der_energy_i+=1/2 * mass_of_particle * (pressure_i/density_i**2 + pressure[j]/State_vector[j,3]**2+visc) * (velocity_i-State_vector[j,4]) * Delta_W_value[i,j]
        
        #add the derivative of density, velocity and energy to the derivative state vector
        State_vector_dir[i,3] = der_density_i
        State_vector_dir[i,4] = der_velocity_i
        State_vector_dir[i,7] = der_energy_i
    
    #set the ghost particles to 0
    ghost_particle = 10
    outside_ghost = (np.arange(number_of_particles) < ghost_particle) | (np.arange(number_of_particles) > (399 - ghost_particle))
    State_vector_dir[outside_ghost, 0:3] = 0
    State_vector_dir[outside_ghost, 3:5] = 0
    State_vector_dir[outside_ghost, 7] = 0
        
    return State_vector_dir

# Set the initial conditions
t=0
h=0.005
t_end=h*10

# Initialize the RK45 integrator
def RK4(State_vector, t, h, G_function):
    k1 = G_function(State_vector, t)
    k2 = G_function(State_vector + 0.5*h*k1, t + 0.5*h)
    k3 = G_function(State_vector + 0.5*h*k2, t + 0.5*h)
    k4 = G_function(State_vector + h*k3, t + h)
    W_next = State_vector + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return W_next


# To store the results
result = np.zeros((int((t_end - t) / h)+1, number_of_particles, len(initial_conditions_x_less_or_equal_0) + 3))

# Set the initial conditions
result[0]=State_vector







# Run the simulation and store the results
from tqdm import tqdm
pbar = tqdm(total=int((t_end - t) / h))
for i, t in enumerate(np.arange(t, t_end, h)):
    pbar.update(1)
    result[i+1]=State_vector
    State_vector = RK4(State_vector, t, h, G_function)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#plot the results of the simulation
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

x = result[:, :, 0]
density = result[:, :, 3]
energy = result[:, :, 7]
gamma = 1.4
pressure = (gamma - 1) * density * energy
velocity_x = result[:, :, 4]

scatters = [ax.scatter([], [],s=5) for ax in axs]

def init():
    for scatter in scatters:
        scatter.set_offsets(np.empty((0, 2))) # use empty 2D array
    axs[0].set_title('density')
    axs[0].set_xlabel('x')
    axs[0].set_xlim([np.min(x), np.max(x)])
    #axs[0].set_ylim([np.min(density), np.max(density)])
    #make the y_lim a bit bigger than the max and min
    axs[0].set_ylim([np.min(density)-0.1, np.max(density)+0.1])
    axs[1].set_title('pressure')
    axs[1].set_xlabel('x')
    axs[1].set_xlim([np.min(x), np.max(x)])
    #axs[1].set_ylim([np.min(pressure), np.max(pressure)])
    axs[1].set_ylim([np.min(pressure)-0.1, np.max(pressure)+0.1])

    axs[2].set_title('velocity_x')
    axs[2].set_xlabel('x')
    axs[2].set_xlim([np.min(x), np.max(x)])
    #axs[2].set_ylim([np.min(velocity_x), np.max(velocity_x)])
    axs[2].set_ylim([np.min(velocity_x)-0.1, np.max(velocity_x)+0.1])


    axs[3].set_title('energy')
    axs[3].set_xlabel('x')
    axs[3].set_xlim([np.min(x), np.max(x)])
    #axs[3].set_ylim([np.min(energy), np.max(energy)])
    axs[3].set_ylim([np.min(energy)-0.1, np.max(energy)+0.1])

    return scatters

def update(frame):
    for i, scatter in enumerate(scatters):
        x_data = x[frame]
        y_data = None
        if i == 0:
            y_data = density[frame]
        elif i == 1:
            y_data = pressure[frame]
        elif i == 2:
            y_data = velocity_x[frame]
        elif i == 3:
            y_data = energy[frame]
        data = np.column_stack((x_data, y_data))
        scatter.set_offsets(data)
        
        
    return scatters

anim = FuncAnimation(fig, update, frames=result.shape[0], init_func=init, blit=True)

plt.show()

#"""

#"""

    