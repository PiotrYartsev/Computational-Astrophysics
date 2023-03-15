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

# increase numpy print options to see entire array
np.set_printoptions(threshold=sys.maxsize)

# set the initial conditions

# open file to read
file = open("C:\\Users\\piotr\\Documents\\GitHub\\Computational-Astrophysics\\Project3\\Planet300.dat", "r")

# read the file
lines = file.readlines()

# close the file
file.close()

# create a numpy array to store the state vectors
State_vector = np.zeros((len(lines), 10))

# loop over each line in the file and fill the state vector array
for i in range(len(lines)):
    line = lines[i].split()
    State_vector[i, 0] = float(line[0])
    State_vector[i, 1] = float(line[1])
    State_vector[i, 2] = float(line[2])
    State_vector[i, 3] = float(line[3])
    State_vector[i, 4] = float(line[4])
    State_vector[i, 5] = float(line[5])
    State_vector[i, 6] = float(line[6])
    State_vector[i, 7] = float(line[7])
    State_vector[i, 8] = float(line[8])
    gamma = 1.4
    density = float(line[7])
    pressure = float(line[8])
    energy = pressure / ((gamma - 1) * density)
    State_vector[i, 9] = energy

# x, y, z, vx, vy, vz, mass, density, pressure, energy
number_of_particles = len(lines)

# gravitational constant
G = 6.67408e-11


# derivative of the kernel function
def W_derivat(dx, h, r):
    a_d = 3 / (2 * np.pi * h ** 3)
    R = r / h
    result = np.zeros_like(dx)

    mask1 = np.logical_and(R >= 0, R <= 1)
    mask2 = np.logical_and(R >= 1, R <= 2)
    result[mask1] = a_d * (-2 + 3 / 2 * R[mask1]) * dx[mask1] / h ** 2
    result[mask2] = -a_d * ((1 / 2) * (2 - R[mask2]) ** 2) * dx[mask2] / (h * r[mask2])
    return result


def Gravity_phi(dx, h, r):
    R = r / h
    result = np.zeros_like(dx)
    mask1 = np.logical_and(R >= 0, R <= 1)
    mask2 = np.logical_and(R >= 1, R <= 2)
    result[mask1] = (1 / h ** 2) * (4 / 3 * R[mask1] - 6 / 3 * R[mask1] ** 3 + 1 / 2 * R[mask1] ** 4)
    result[mask2] = (1 / h ** 2) * (
            8 / 3 * R[mask2] - 3 * R[mask2] ** 2 + 6 / 5 * R[mask2] ** 3 - 1/6  *R[mask2]**4-1/(15*R[mask2]**2))
    #else:
    result[~np.logical_or(mask1, mask2)] = (1/r[~np.logical_or(mask1, mask2)]**2)
    #result[~np.logical_or(mask1, mask2)] = (1/r[~np.logical_or(mask1, mask2)]**2)
    return result


def visc(density_ij, c_ij, h, v, r):
    
    phistuff = 0.1 * h
    
    # dot product of the velocity and position
    v_dot_r = np.sum(v*r, axis=1)
    
    # absolute value of r
    r_abs = np.linalg.norm(r, axis=1)
    
    theta = np.zeros_like(r)
    
    mask = np.where(v_dot_r < 0)
    theta[mask] = (h * v_dot_r[mask] / r_abs[mask]**2 + phistuff**2)
    
    alpha = 1
    beta = 1
    
    result = (alpha * c_ij * theta + beta * theta**2) / density_ij
    
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
    #duplicate r to make it the same shape as dr
    r = np.repeat(r[np.newaxis, :], 3, axis=0)

    v=np.linalg.norm(dv, axis=0)
    #duplicate r to make it the same shape as dr
    v = np.repeat(v[np.newaxis, :], 3, axis=0)


    #define the derivative of the kernel function
    Delta_W_value = np.zeros((number_of_particles, number_of_particles,3))

    #calculate the kernel function and the derivative of the kernel function
    h_constant = 1e7

    # calculate the smoothing kernel and its derivative
    #W_value = W_kernel(r, h_constant)
    Delta_W_value = W_derivat(dr, h_constant, r)

    # calculate the pressure term
    Pressure_ij = pressure[:, np.newaxis] + pressure
    Pressure_term = np.zeros_like(dr)
    for i in range(number_of_particles):
        Pressure_term[:, i, :] = -mass_of_particle * Pressure_ij[:, i] * Delta_W_value[:, i, :] / rho[:, i]**2

    # calculate the viscosity term
    Visco_term = np.zeros_like(dr)
    for i in range(number_of_particles):
        Visco_term[:, i, :] = mass_of_particle * visc(rho, c, h_constant, v, r)[:, i, :] / rho[:, i]**2

    # calculate the gravity term
    Gravity_term = np.zeros_like(dr)
    for i in range(number_of_particles):
        Gravity_term[:, i, :] = G * mass_of_particle * mass_of_particle[:, np.newaxis] / r[:, i, :]**2 * Gravity_phi[:, i, :]

    # calculate the derivative of the position, velocity, mass, density, pressure, and energy
    Derivative_x_values = velocity_x
    Derivative_y_values = velocity_y
    Derivative_z_values = velocity_z

    Derivative_velocity_x = (Pressure_term[:, :, 0] + Visco_term[:, :, 0] + Gravity_term[:, :, 0]) / density
    Derivative_velocity_y = (Pressure_term[:, :, 1] + Visco_term[:, :, 1] + Gravity_term[:, :, 1]) / density
    Derivative_velocity_z = (Pressure_term[:, :, 2] + Visco_term[:, :, 2] + Gravity_term[:, :, 2]) / density

    Derivative_mass_of_particle = 0

    Derivative_density = np.zeros_like(density)
    for i in range(number_of_particles):
        Derivative_density[i] = np.sum(mass_of_particle * v[:, i] * Delta_W_value[:, i] / h_constant)

    Derivative_pressure = np.zeros_like(pressure)
    for i in range(number_of_particles):
        Derivative_pressure[i] = np.sum(mass_of_particle * (energy + pressure) / rho * v[:, i] * Delta_W_value[:, i] / h_constant)

    Derivative_energy = np.zeros_like(energy)
    for i in range(number_of_particles):
        Derivative_energy[i] = np.sum(mass_of_particle * v[:, i] * (Pressure_term[:, i] + Visco_term[:, i] + Gravity_term[:, i]) / rho[:, i]**2)

    # store the derivatives in the state vector direction
    State_vector_dir[:, 0] = Derivative_x_values
    State_vector_dir[:, 1] = Derivative_y_values
    State_vector_dir[:, 2] = Derivative_z_values
    State_vector_dir[:, 3] = Derivative_velocity_x
    State_vector_dir[:, 4] = Derivative_velocity_y
    State_vector_dir[:, 5] = Derivative_velocity_z
    State_vector_dir[:, 6] = Derivative_mass_of_particle
    State_vector_dir[:, 7] = Derivative_density
    State_vector_dir[:, 8] = Derivative_pressure
    State_vector_dir[:, 9] = Derivative_energy

    return State_vector_dir.flatten()

Delta_W_value=(G_function(State_vector,0))
#"""