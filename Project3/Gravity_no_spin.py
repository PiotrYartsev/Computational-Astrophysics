from tqdm import tqdm
import math as math
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm 
import sys
import numpy
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
numpy.set_printoptions(threshold=sys.maxsize)
#set the initial conditions

#open file to read
file=open("C:\\Users\\piotr\\Documents\\GitHub\\Computational-Astrophysics\\Project3\\Planet300.dat","r")
#read the file
lines=file.readlines()
#close the file
file.close()

State_vector=np.zeros((len(lines),10))
for i in range(len(lines)):
    #print(line)
    line=lines[i]
    line=line.split()
    State_vector[i,0]=float(line[0])
    State_vector[i,1]=float(line[1])
    State_vector[i,2]=float(line[2])
    State_vector[i,3]=float(line[3])
    State_vector[i,4]=float(line[4])
    State_vector[i,5]=float(line[5])
    State_vector[i,6]=float(line[6])
    State_vector[i,7]=float(line[7])
    State_vector[i,8]=float(line[8])
    gamma = 1.4
    density = float(line[7])
    pressure = float(line[8])
    energy = pressure/((gamma - 1) * density)
    State_vector[i,9]=energy

#x,y,z,vx,vy,vz,mass, density, pressure, energy
number_of_particles=len(lines)

#kernel functions
def W(dx, h):
    a_d = 3 / (2 * np.pi * h**3)
    r = np.linalg.norm(dx, axis=1)
    R = r / h
    result = np.zeros(len(dx))
    mask1 = np.logical_and(R >= 0, R <= 1)
    mask2 = np.logical_and(R >= 1, R <= 2)
    result[mask1] = a_d * (2/3 - R[mask1]**2 + 1/2 * R[mask1]**3)
    result[mask2] = a_d * (1/6 * (2-R[mask2])**3)
    return result

#derivative of the kernel function
def smoothingdW(r, dX, hmean):

    ad = (3/(2*np.pi*hmean**3)) # Alpha-d factor
    R = r/hmean
    smoothdW = np.zeros((3, len(R))) 
    
    # Define masks
    mask_01 = (R >= 0) & (R < 1) # First condition
    mask_02 = (R >= 1) & (R < 2) # Second condition
      
    # Stack individual masked vectors into arrays
    dX_1 = dX[:,mask_01]
    dX_2 = dX[:,mask_02]
    
    # Calculate all values for the derivatives of the smoothing function given the conditions
    constant1 = ad*(-2 + 1.5*(R[mask_01]))/(hmean**2)    
    smoothdW[:,mask_01] = constant1*(dX_1)
    
    constant2 = -ad*(0.5*((2-(R[mask_02]))**2))/(hmean*r[mask_02])    
    smoothdW[:,mask_02] = constant2*(dX_2)

    
    return smoothdW

def phi_derivative(dx, h):
    r = np.linalg.norm(dx, axis=1)
    R = r / h
    result = np.zeros(len(dx))
    mask1 = np.logical_and(R >= 0, R <= 1)
    mask2 = np.logical_and(R >= 1, R <= 2)
    result[mask1] = (1/h**2)*(4/3 * R[mask1] - 6/3 *R[mask1]**3+ 1/2  * R[mask1]**4)
    result[mask2] = (1/h**2)*(8/3 * R[mask2] - 3*R[mask2]**2   +   6/5  *R[mask2]**3-   1/6  *R[mask2]**4-1/(15*R[mask2]**2))
    result[~np.logical_or(mask1, mask2)] = (1/r[~np.logical_or(mask1, mask2)]**2)
    return result
    

def G_function(State_vector,t):
    State_vector_dir = np.zeros((number_of_particles, 10))
    x_values=State_vector[:,0]
    y_values=State_vector[:,1]
    z_values=State_vector[:,2]
    velocity_x=State_vector[:,3]
    velocity_y=State_vector[:,4]
    velocity_z=State_vector[:,5]
    mass_of_particle=State_vector[:,6]
    density=State_vector[:,7]
    pressure=State_vector[:,8]
    energy=State_vector[:,9]

    Derivative_x_values=State_vector_dir[:,0]
    Derivative_y_values=State_vector_dir[:,1]
    Derivative_z_values=State_vector_dir[:,2]
    Derivative_velocity_x=State_vector_dir[:,3]
    Derivative_velocity_y=State_vector_dir[:,4]
    Derivative_velocity_z=State_vector_dir[:,5]
    Derivative_mass_of_particle=State_vector_dir[:,6]
    Derivative_density=State_vector_dir[:,7]
    Derivative_pressure=State_vector_dir[:,8]
    Derivative_energy=State_vector_dir[:,9]

    
    #calculate the distance between the particles
    dx= x_values[:, np.newaxis] - x_values
    dy= y_values[:, np.newaxis] - y_values
    dz= z_values[:, np.newaxis] - z_values

    dvx= velocity_x[:, np.newaxis] - velocity_x
    dvy= velocity_y[:, np.newaxis] - velocity_y
    dvz= velocity_z[:, np.newaxis] - velocity_z

    dr=np.array((dx,dy,dz))
    dv=np.array((dvx,dvy,dvz))
    print(dr.shape)
    r=np.linalg.norm(dr, axis=0)
    print(r.shape)
    #define the derivative of the kernel function
    Delta_W_value = np.zeros((number_of_particles, number_of_particles,3))

    #calculate the kernel function and the derivative of the kernel function
    h_constant = 1e7    
    Delta_W_value = smoothingdW(r, dr, h_constant)
    return(Delta_W_value)

print(G_function(State_vector,0))
#"""
"""
    
    
    #set the 
    State_vector_dir[:,0]=Derivative_x_values
    State_vector_dir[:,1]=Derivative_y_values
    State_vector_dir[:,2]=Derivative_z_values
    State_vector_dir[:,3]=0
    State_vector_dir[:,4]=Derivative_velocity_x
    State_vector_dir[:,5]=Derivative_velocity_y
    State_vector_dir[:,6]=Derivative_velocity_z
    State_vector_dir[:,7]=Derivative_energy

    State_vector[:,3]=density

    return State_vector_dir

# Set the initial conditions
t=0
h=0.005
t_end=h*40

# Initialize the RK45 integrator
def RK4(State_vector, t, h, G_function):
    f1 = G_function(State_vector, t)
    f2= G_function(State_vector + f1*h/2, t + h/2)
    f3= G_function(State_vector + f2*h/2, t + h/2)
    f4= G_function(State_vector + f3*h, t + h)
    W_next = State_vector + h*(f1 + 2*f2 + 2*f3 + f4)/6
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
y = result[:, :, 1]
z = result[:, :, 2]
density = result[:, :, 3]
velocity_x = result[:, :, 4]
velocity_y = result[:, :, 5]
velocity_z = result[:, :, 6]
energy = result[:, :, 7]

gamma = 1.4
pressure = (gamma - 1) * density * energy


scatters = [ax.scatter([], [],s=5) for ax in axs]

def init():
    for scatter in scatters:
        scatter.set_offsets(np.empty((0, 2))) # use empty 2D array

    #set title for entire figure
    fig.suptitle('1D SPH simulation, no arteficial viscosity', fontsize=16)
    axs[0].set_title('Density')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Density (Kg/m^3)')
    axs[0].set_xlim([-0.6, 0.6])
    axs[0].set_ylim([np.min(density)-0.1, np.max(density)+0.1])

    axs[1].set_title('Pressure')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Pressure (N/m^3)')
    axs[1].set_xlim([-0.6, 0.6])
    #axs[1].set_ylim([np.min(pressure), np.max(pressure)])
    axs[1].set_ylim([np.min(pressure)-0.1, np.max(pressure)+0.1])

    axs[2].set_title('velocity(x)')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Velocity(x) (m/s)')
    axs[2].set_xlim([-0.6, 0.6])
    #axs[2].set_ylim([np.min(velocity_x), np.max(velocity_x)])
    axs[2].set_ylim([np.min(velocity_x)-0.1, np.max(velocity_x)+0.1])


    axs[3].set_title('Internal energy')
    axs[3].set_xlabel('x')
    axs[3].set_ylabel('Internal energy (J/Kg)')
    axs[3].set_xlim([-0.6, 0.6])
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

#anim = FuncAnimation(fig, update, frames=result.shape[0], init_func=init, blit=True)

#animate at half the speed
anim = FuncAnimation(fig, update, frames=result.shape[0], init_func=init, blit=True, interval=90)
writer=animation.FFMpegWriter(fps=3,extra_args=['-vcodec', 'libx264'])
anim.save('no_visc.mp4',writer=writer)

plt.close()

np.savetxt("no_visc.csv", result[-1], delimiter=",")

#"""