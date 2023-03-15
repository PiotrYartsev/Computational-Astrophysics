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


G = 6.67408e-11


#derivative of the kernel function
def W_derivat(dx, h, r):
    #make r into a 3D vector of r,r,r
    r=np.array([r,r,r])

    a_d = 3 / (2 * np.pi * h**3)
    R = r / h    
    

    result = np.zeros_like(dx)
    
    mask1 = np.logical_and(R >= 0, R <= 1)
    mask2 = np.logical_and(R >= 1, R <= 2)
    result[mask1] = a_d * (-2 + 3/2 * R[mask1]) * dx[mask1] / h**2
    result[mask2] = -a_d * ((1/2) * (2 - R[mask2])**2) * dx[mask2] / (h * r[mask2])

    result= result.transpose((1,2,0))
    return result


def gravity(dx, h, r):
    r = np.array([r, r, r])
    R = r / h
    result = np.zeros_like(dx)
    mask1 = np.logical_and(R >= 0, R <= 1)
    mask2 = np.logical_and(R >= 1, R <= 2)
    #if R>2 
    mask3 = R>2
    result[mask1] = (1/h**2)*(4/3 * R[mask1] - 6/3 * R[mask1]**3 + 1/2 * R[mask1]**4)
    result[mask2] = (1/h**2)*(8/3 * R[mask2] - 3*R[mask2]**2 + 6/5 * R[mask2]**3 - 1/6 * R[mask2]**4 - 1/(15*R[mask2]**2))
    #if not mask1 or mask2:
    result[mask3] = 1/(r[mask3]**2)
    
    # Swap the first and last dimensions
    result = result.transpose((1, 2, 0))
    
    return result


def visc(density_ij, c_ij, h, v, r):
    r = np.array([r, r, r]).T
    v= np.array([v, v, v]).T
    c_ij = np.array([c_ij, c_ij, c_ij]).T
    density_ij = np.array([density_ij, density_ij, density_ij]).T
    
    phistuff = 0.1 * h
    
    # dot product of the velocity and position
    v_dot_r = np.sum(v * r, axis=1)
    
    # absolute value of r
    r_abs = np.linalg.norm(r, axis=1)
    
    theta = np.zeros_like(r_abs)

    mask = np.where(v_dot_r < 0)
    theta[mask] = (h * v_dot_r[mask] / r_abs[mask]**2 + phistuff**2)
    
    alpha = 1
    beta = 1
    
    result = (alpha * c_ij * theta + beta * theta**2) / density_ij
    
    return result


    

def G_function(State_vector,t):
    #print("State vector",State_vector.shape)
    #State_vector_dir the same shape as State_vector
    State_vector_dir = np.zeros_like(State_vector)
    #print("State vector dir",State_vector_dir.shape)
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

    
    speed_of_sound=np.sqrt((gamma-1)*energy)

    #make a matrix c where c[i,j] is the average speed of sound of particle i and j
    c = np.zeros((number_of_particles, number_of_particles))
    c = (speed_of_sound[:, np.newaxis] + speed_of_sound) / 2

    #make a matrix rho where rho[i,j] is the average density of particle i and j
    rho = np.zeros((number_of_particles, number_of_particles))
    rho = (density[:, np.newaxis] + density) / 2



    #calculate the distance between the particles
    dx= x_values[:, np.newaxis] - x_values
    dy= y_values[:, np.newaxis] - y_values
    dz= z_values[:, np.newaxis] - z_values

    dvx= velocity_x[:, np.newaxis] - velocity_x
    dvy= velocity_y[:, np.newaxis] - velocity_y
    dvz= velocity_z[:, np.newaxis] - velocity_z

    dr=np.array((dx,dy,dz))
    dv=np.array((dvx,dvy,dvz))

    r=np.linalg.norm(dr, axis=0)
    #print("r",r.shape)


    v=np.linalg.norm(dv, axis=0)
    #print("v",v.shape)



    #define the derivative of the kernel function
    Delta_W_value = np.zeros((number_of_particles, number_of_particles,3))

    #calculate the kernel function and the derivative of the kernel function
    h_constant = 1e7    
    Delta_W_value = W_derivat(dr, h_constant, r)
    #print("Delta_W_value",Delta_W_value.shape)

    Gravity_phi = np.zeros((number_of_particles, number_of_particles,3))
    Gravity_phi = gravity(dr, h_constant,r)
    #print("Gravity_phi",Gravity_phi.shape)
    
    Visco = np.zeros((number_of_particles, number_of_particles,3))
    Visco = visc(rho,c,h_constant,v,r)
    #print("Visco",Visco.shape)

    #define the pressure 
    pressure = np.zeros(number_of_particles)
    pressure = (gamma-1)*density*energy
    State_vector[:,8]=pressure

    #start the calculation of the derivatives
    #calculate the derivative of the x values
    Derivative_x_values = velocity_x
    #calculate the derivative of the y values
    Derivative_y_values = velocity_y
    #calculate the derivative of the z values
    Derivative_z_values = velocity_z

    for i in range(number_of_particles):
        density_i=density[i]*np.ones(number_of_particles)
        #same for pressure
        pressure_i=pressure[i]*np.ones(number_of_particles)
        #same for velocity
        velocity_y_i=velocity_y[i]*np.ones(number_of_particles)
        velocity_x_i=velocity_x[i]*np.ones(number_of_particles)
        velocity_z_i=velocity_z[i]*np.ones(number_of_particles)
        v_i=np.array((velocity_x_i,velocity_y_i,velocity_z_i))
        v_1_v_j=v_i[:,np.newaxis]-dv
        #calculate the derivative of the velocity in the x direction
        Derivative_velocity_x[i] = -np.sum(mass_of_particle*((pressure_i/(density_i**2))+(pressure/(density**2))+Visco[i,:,0])*Delta_W_value[i,:,0])-G*np.sum(mass_of_particle*Gravity_phi[i,:,0])
        #calculate the derivative of the velocity in the y direction
        Derivative_velocity_y[i] = -np.sum(mass_of_particle*((pressure_i/(density_i**2))+(pressure/(density**2))+Visco[i,:,1])*Delta_W_value[i,:,1])-G*np.sum(mass_of_particle*Gravity_phi[i,:,1])
        #calculate the derivative of the velocity in the z direction
        Derivative_velocity_z[i] = -np.sum(mass_of_particle*((pressure_i/(density_i**2))+(pressure/(density**2))+Visco[i,:,2])*Delta_W_value[i,:,2])-G*np.sum(mass_of_particle*Gravity_phi[i,:,2])
        #calculate the derivative of the mass of the particle
        Derivative_mass_of_particle[i] = 0
        #calculate the derivative of the density
        Derivative_density[i] = np.sum(mass_of_particle*np.sum((v_1_v_j)*Delta_W_value[i,:,0],axis=0))
        #calculate the derivative of the pressure
        Derivative_pressure[i] = 0
        #calculate the derivative of the energy
        Derivative_energy[i] = 1/2*np.sum(mass_of_particle*((pressure_i/(density_i**2))+(pressure/(density**2))+Visco[i,:,0])*np.sum((v_1_v_j)*Delta_W_value[i,:,0],axis=0))

        if Derivative_energy[i] < 0:
            Derivative_energy[i] = 0

    State_vector_dir[:,0]=Derivative_x_values
    State_vector_dir[:,1]=Derivative_y_values
    State_vector_dir[:,2]=Derivative_z_values
    State_vector_dir[:,3]=Derivative_velocity_x
    State_vector_dir[:,4]=Derivative_velocity_y
    State_vector_dir[:,5]=Derivative_velocity_z
    State_vector_dir[:,6]=Derivative_mass_of_particle
    State_vector_dir[:,7]=Derivative_density
    State_vector_dir[:,8]=Derivative_pressure
    State_vector_dir[:,9]=Derivative_energy

    return State_vector_dir
    

Delta_W_value=(G_function(State_vector,0))


# Set the initial conditions
t=0
h=100
t_end=h*50

# Initialize the RK45 integrator
def RK4(State_vector, t, h, G_function):
    f1 = G_function(State_vector, t)
    f2= G_function(State_vector + f1*h/2, t + h/2)
    f3= G_function(State_vector + f2*h/2, t + h/2)
    f4= G_function(State_vector + f3*h, t + h)
    W_next = State_vector + h*(f1 + 2*f2 + 2*f3 + f4)/6
    return W_next


# To store the results
result = np.zeros((int((t_end - t) / h)+1, number_of_particles, 10))

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
"""
#plot the first frame in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x=result[0,:,0]
y=result[0,:,1]
z=result[0,:,2]
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
plt.close()"""


#animate it in 3D
t=0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = result[:,:,0]
y = result[:,:,1]
z = result[:,:,2]

#set permanent axis limits


def animate(i):
    ax.set_xlim3d(np.min(x), np.max(x))
    ax.set_ylim3d(np.min(y), np.max(y))
    ax.set_zlim3d(np.min(z), np.max(z))
    ax.view_init(elev=20, azim=30) 
    ax.scatter(x[i], y[i], z[i], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('Simulation at t = {:.2f}'.format(t + i * h))
    #adjust the viewing angle
    ax.view_init(elev=20, azim=30)

ani = FuncAnimation(fig, animate, frames=range(0, len(x)), interval=1)
plt.show()