from tqdm.notebook import tqdm
import math as math
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm 
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
#set the initial conditions

# density, velocity, pressure, eneergy, distance between particles
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
for i in np.linspace(0+small_step,0.6+small_step,80):
    State_vector.append((i, 0, 0, initial_conditions_x_greater_0[0], initial_conditions_x_greater_0[1], initial_conditions_x_greater_0[2], initial_conditions_x_greater_0[3], initial_conditions_x_greater_0[4]))
State_vector=np.array(State_vector)
x=State_vector[:,0]
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
    r=norm(dx)
    R=r/h
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
    State_vector_dir = np.zeros((400, len(initial_conditions_x_less_or_equal_0) + 3))


    #just use a defoult value
    h_1=0.002
    h_2=h_1*5
    
    h_list=[]
    for i in range(320):
        h_list.append(h_1)
    for i in range(320,400):
        h_list.append(h_2)
    #print(h_list)

    k=2
    dx = State_vector[:, 0] - State_vector[:, 0][:, np.newaxis]

    W_value = np.zeros((400, 400))
    Delta_W_value = np.zeros((400, 400))

    for i in range(400):
        for j in range(400):
            h=(h_list[i]+h_list[j])/2
            if norm(dx[i, j])<=(k*(h)) and i!=j:
                W_value[i, j] = W(dx[i,j], h)
                Delta_W_value[i, j] = W_derivat(dx[i, j], h)

    
    State_vector_dir[:, 0] = State_vector[:, 4]
    State_vector_dir[:, 1] = State_vector[:, 5]
    State_vector_dir[:, 2] = State_vector[:, 6]

    gamma = 1.4
    pressure = (gamma - 1) * State_vector[:, 3] * State_vector[:, 7]
    seed_of_sound = np.sqrt((gamma - 1) * State_vector[:, 7])
    for i in range(400):
        #make pressure_i the same chape as the whole pressure array
        pressure_i = pressure[i] * np.ones(400)
        #same with dentsity
        density_i = State_vector[i, 3] * np.ones(400)
        #same for velocity
        velocity_i = State_vector[i, 4] * np.ones(400)
        #density
        State_vector_dir[i, 3] = np.sum(mass_of_particle *(velocity_i-State_vector[:,4])* W_value[i], axis=0)
        #velocity
        State_vector_dir[i, 4] = -np.sum(mass_of_particle * (pressure_i/density_i**2 + pressure/State_vector[:,3]+visc) * Delta_W_value[i], axis=0)
        #energy
        State_vector_dir[i,7]=1/2 * np.sum(mass_of_particle * (pressure_i/density_i**2 + pressure/State_vector[:,3]+visc) * (velocity_i-State_vector[:,4]) * Delta_W_value[i], axis=0)
        


    
    return State_vector_dir

# Set the initial conditions

t=0
h=0.005
t_end=h*2

# Initialize the RK45 integrator
def RK4(State_vector, t, h, G_function):
    k1 = G_function(State_vector, t)
    k2 = G_function(State_vector + 0.5*h*k1, t + 0.5*h)
    k3 = G_function(State_vector + 0.5*h*k2, t + 0.5*h)
    k4 = G_function(State_vector + h*k3, t + h)
    W_next = State_vector + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return W_next


# To store the results
result = np.zeros((int((t_end - t) / h)+1, 400, len(initial_conditions_x_less_or_equal_0) + 3))
result[0]=State_vector
# Iterate over time steps using tqdm to display a progress bar
from tqdm import tqdm
pbar = tqdm(total=int((t_end - t) / h))
for i, t in enumerate(np.arange(t, t_end, h)):
    pbar.update(1)
    result[i+1]=State_vector
    State_vector = RK4(State_vector, t, h, G_function)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig, axs = plt.subplots(1, 4, figsize=(20, 5))

x = result[:, :, 0]
density = result[:, :, 3]
energy = result[:, :, 7]
gamma = 1.4
pressure = (gamma - 1) * density * energy
velocity_x = result[:, :, 4]

for i in range(len(x)):
    #print largest and smallest x
    print(np.max(x[i]), np.min(x[i]))

lines = [ax.plot([], [])[0] for ax in axs]

def init():
    for line in lines:
        line.set_data([], [])
    axs[0].set_title('density')
    axs[1].set_title('pressure')
    axs[2].set_title('velocity_x')
    axs[3].set_title('energy')
    return lines

def update(frame):
    for i, line in enumerate(lines):
        #order x and make averythign else follow the same order
        if i == 0:
            line.set_data(x[frame], density[frame])
            axs[i].set_xlim([np.min(x), np.max(x)])
            axs[i].set_ylim([np.min(density), np.max(density)])
        elif i == 1:
            line.set_data(x[frame], pressure[frame])
            axs[i].set_xlim([np.min(x), np.max(x)])
            axs[i].set_ylim([np.min(pressure), np.max(pressure)])
        elif i == 2:
            line.set_data(x[frame], velocity_x[frame])
            axs[i].set_xlim([np.min(x), np.max(x)])
            axs[i].set_ylim([np.min(velocity_x), np.max(velocity_x)])
        elif i == 3:
            line.set_data(x[frame], energy[frame])
            axs[i].set_xlim([np.min(x), np.max(x)])
            axs[i].set_ylim([np.min(energy), np.max(energy)])
    return lines

anim = FuncAnimation(fig, update, frames=result.shape[0], init_func=init, blit=True)

plt.show()

#"""