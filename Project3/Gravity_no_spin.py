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
file=open("C:\\Users\\MSI PC\\Desktop\\Computational-Astrophysics\\Project3\\Planet300.dat","r")
#read the file
lines=file.readlines()
#close the file
file.close()

State_vector=np.zeros((len(lines),9))
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



#print(State_vector)
#x,y,z,vx,vy,vz,mass, density, pressure, energy
number_of_particles=len(lines)

#kernel functions
def W(dx,h):
    a_d=3/(2*math.pi*h**3)
    r=norm(dx)
    R=r/h
    if R<=1 and R>=0:
        return a_d * (2/3 - R**2 + 1/2 * R**3)
    else:
        return a_d * (1/6 * (2-R)**3)


#derivative of the kernel function
def W_derivat(dx,h):
    a_d=3/(2*math.pi*h**3)
    #print("a_d",a_d)
    r=norm(dx)
    #print("r",r)
    R=r/h
    #print("R",R)
    if R<=1 and R>=0:
        return a_d * (-2 + 3/2 * R) * dx / h**2
    else:
        return -a_d * ((1/2) * (2 - R)**2) * dx / (h * r)
    
def phi_derivative(dx,h):
    r=norm(dx)
    #print("r",r)
    R=r/h
    #print("R",R)
    if R<=1 and R>=0:
        return (1/h**2)*(4/3 * R - 6/3 *R**3+ 1/2  * R**4)
    elif R>=1 and R<=2:
        return (1/h**2)*(8/3 * R - 3*R**2   +   6/5  *R**3-   1/6  *R**4-1/(15*R**2))
    else:
        return(1/r**2)
    

def G_function(State_vector,t):
    State_vector_dir = np.zeros((number_of_particles, 9))
    x_values=State_vector[:,0]
    y_values=State_vector[:,1]
    z_values=State_vector[:,2]
    velocity_x=State_vector[:,3]
    velocity_y=State_vector[:,4]
    velocity_z=State_vector[:,5]
    mass_of_particle=State_vector[:,6]
    density=State_vector[:,7]
    pressure=State_vector[:,8]

    Derivative_x_values=State_vector_dir[:,0]
    Derivative_y_values=State_vector_dir[:,1]
    Derivative_z_values=State_vector_dir[:,2]
    Derivative_velocity_x=State_vector_dir[:,3]
    Derivative_velocity_y=State_vector_dir[:,4]
    Derivative_velocity_z=State_vector_dir[:,5]
    Derivative_mass_of_particle=State_vector_dir[:,6]
    Derivative_density=State_vector_dir[:,7]
    Derivative_pressure=State_vector_dir[:,8]


    #create a array of h values with length of the number of particles
    d=3
    h_list=1.3*(mass_of_particle/density)**(1/d)


    #calculate the distance between the particles
    dx= x_values[:, np.newaxis] - x_values
    dy= y_values[:, np.newaxis] - y_values
    dz= z_values[:, np.newaxis] - z_values
    dvx= velocity_x[:, np.newaxis] - velocity_x
    dvy= velocity_y[:, np.newaxis] - velocity_y
    dvz= velocity_z[:, np.newaxis] - velocity_z

    #define the kernel function
    #W_value = np.zeros((number_of_particles, number_of_particles,3))

    #define the derivative of the kernel function
    Delta_W_value = np.zeros((number_of_particles, number_of_particles,3))

    #define the gravitational potential
    phi_value = np.zeros((number_of_particles, number_of_particles,3))

    #calculate the kernel function and the derivative of the kernel function
    k=2
    for i in tqdm(range(number_of_particles)):
        for j in range(number_of_particles):
            h_ij=(h_list[i]+h_list[j])/2
            phi_value[i, j,0] = phi_derivative(dx[i, j], h_ij)
            phi_value[i, j,1] = phi_derivative(dy[i, j], h_ij)
            phi_value[i, j,2] = phi_derivative(dz[i, j], h_ij)
            if norm(dx[i, j])<=(k*(h_ij)):
                #W_value[i, j,0] = W(dx[i,j], h_ij)
                #W_value[i, j,1] = W(dy[i,j], h_ij)
                #W_value[i, j,2] = W(dz[i,j], h_ij)
                Delta_W_value[i, j,0] = W_derivat(dx[i, j], h_ij)
                Delta_W_value[i, j,1] = W_derivat(dy[i, j], h_ij)
                Delta_W_value[i, j,2] = W_derivat(dz[i, j], h_ij)

    print("pressure",pressure)

    visc=np.zeros((number_of_particles, number_of_particles))


    Derivative_x_values = velocity_x
    Derivative_y_values = velocity_y
    Derivative_z_values = velocity_z
    for i in range(number_of_particles):
        if x_values[i]<-0.6 or x_values[i]>0.6:
            pass
        else:
            #make a list of density_i the same length as the number of particles
            density_i=density[i]*np.ones(number_of_particles)
            #same for pressure
            pressure_i=pressure[i]*np.ones(number_of_particles)
            #same for velocity
            velocity_y_i=velocity_y[i]*np.ones(number_of_particles)
            velocity_x_i=velocity_x[i]*np.ones(number_of_particles)
            velocity_z_i=velocity_z[i]*np.ones(number_of_particles)

            #calculate the derivative of the velocity
            Derivative_velocity_x[i]=np.sum(-mass_of_particle*(pressure_i/density_i**2+pressure/density**2+visc[i,:])*Delta_W_value[i,:,0])
            Derivative_velocity_y[i]=np.sum(-mass_of_particle*(pressure_i/density_i**2+pressure/density**2+visc[i,:])*Delta_W_value[i,:,1])
            Derivative_velocity_z[i]=np.sum(-mass_of_particle*(pressure_i/density_i**2+pressure/density**2+visc[i,:])*Delta_W_value[i,:,2])

            

    return(h_list)

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