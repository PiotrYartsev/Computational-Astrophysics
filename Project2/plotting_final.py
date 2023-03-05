from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


#open file for reading


no_visc = open('/home/piotr/Documents/GitHub/Computational-Astrophysics/no_visc.csv', 'r')
visc = open('/home/piotr/Documents/GitHub/Computational-Astrophysics/with_visc.csv', 'r')


#read data from file
no_visc_data = no_visc.readlines()
visc_data = visc.readlines()

number_of_lines = len(no_visc_data)
number_of_columns = len(no_visc_data[0].split(','))
result_no_visc=np.zeros((number_of_lines, number_of_columns))
result_visc=np.zeros((number_of_lines, number_of_columns))

for i in range(len(no_visc_data)):
    result_no_visc[i] = no_visc_data[i].split(',')
    result_visc[i] = visc_data[i].split(',')


fig, axs = plt.subplots(1, 4, figsize=(20, 5))

x_no_visc = result_no_visc[:, 0]
y_no_visc = result_no_visc[:, 1]
z_no_visc = result_no_visc[:, 2]
density_no_visc = result_no_visc[:, 3]
velocity_x_no_visc = result_no_visc[:, 4]
velocity_y_no_visc = result_no_visc[:, 5]
velocity_z_no_visc = result_no_visc[:, 6]
energy_no_visc = result_no_visc[:, 7]

gamma = 1.4
pressure_no_visc = (gamma - 1) * density_no_visc * energy_no_visc

x_visc = result_visc[:, 0]
y_visc = result_visc[:, 1]
z_visc = result_visc[:, 2]
density_visc = result_visc[:, 3]
velocity_x_visc = result_visc[:, 4]
velocity_y_visc = result_visc[:, 5]
velocity_z_visc = result_visc[:, 6]
energy_visc = result_visc[:, 7]

pressure_visc = (gamma - 1) * density_visc * energy_visc


#plot x vs density
axs[0].plot(x_no_visc, density_no_visc, label='no viscosity')
axs[0].plot(x_visc, density_visc, label='with viscosity')

#plot x vs pressure
axs[1].plot(x_no_visc, pressure_no_visc, label='no viscosity')
axs[1].plot(x_visc, pressure_visc, label='with viscosity')

#plot x vs velocity
axs[2].plot(x_no_visc, velocity_x_no_visc, label='no viscosity')
axs[2].plot(x_visc, velocity_x_visc, label='with viscosity')

#plot x vs energy
axs[3].plot(x_no_visc, energy_no_visc, label='no viscosity')
axs[3].plot(x_visc, energy_visc, label='with viscosity')

#set labels
fig.suptitle('1D SPH simulation, with artificial viscosity', fontsize=16)
axs[0].set_title('Density')
axs[0].set_xlabel('x')
axs[0].set_ylabel('Density (Kg/m^3)')
axs[0].set_xlim([-0.6, 0.6])
axs[0].set_ylim([np.min(density_no_visc)-0.1, np.max(density_no_visc)+0.1])

axs[1].set_title('Pressure')
axs[1].set_xlabel('x')
axs[1].set_ylabel('Pressure (N/m^3)')
axs[1].set_xlim([-0.6, 0.6])
axs[1].set_ylim([np.min(pressure_no_visc)-0.1, np.max(pressure_no_visc)+0.1])

axs[2].set_title('velocity(x)')
axs[2].set_xlabel('x')
axs[2].set_ylabel('Velocity(x) (m/s)')
axs[2].set_xlim([-0.6, 0.6])
axs[2].set_ylim([np.min(velocity_x_no_visc)-0.1, np.max(velocity_x_no_visc)+0.1])


axs[3].set_title('Internal energy')
axs[3].set_xlabel('x')
axs[3].set_ylabel('Internal energy (J/Kg)')
axs[3].set_xlim([-0.6, 0.6])
axs[3].set_ylim([np.min(energy_no_visc)-0.1, np.max(energy_no_visc)+0.1])

#set legend
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[3].legend()

plt.show()