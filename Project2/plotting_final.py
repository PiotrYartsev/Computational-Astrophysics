from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


#open file for reading


no_visc = open('/home/piotr/Documents/GitHub/Computational-Astrophysics/with_visc.csv', 'r')
visc = open('/home/piotr/Documents/GitHub/Computational-Astrophysics/no_visc.csv', 'r')
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

fig.suptitle('1D SPH simulation, with and without artificial viscosity', fontsize=16)

list_of_files = [no_visc, visc]
x1=[]
y1=[]
z1=[]
density1=[]
velocity_x1=[]
velocity_y1=[]
velocity_z1=[]
energy1=[]
pressure1=[]
for file in list_of_files:
    lines=file.readlines()
    x=[]
    y=[]
    z=[]
    density=[]
    velocity_x=[]
    velocity_y=[]
    velocity_z=[]
    energy=[]
    for i in lines:
        data=i.split(',')
        x.append(float(data[0]))
        y.append(float(data[1]))
        z.append(float(data[2]))
        density.append(float(data[3]))
        velocity_x.append(float(data[4]))
        velocity_y.append(float(data[5]))
        velocity_z.append(float(data[6]))
        energy.append(float(data[7]))

    gamma = 1.4
    pressure = []
    for i in range(len(density)):
        pressure.append((gamma-1)*density[i] * energy[i])
    #plot scatter of x vs density, pressure, velocity and energy
    #
    x1.append(x)
    y1.append(y)
    z1.append(z)
    density1.append(density)
    velocity_x1.append(velocity_x)
    velocity_y1.append(velocity_y)
    velocity_z1.append(velocity_z)
    energy1.append(energy)
    pressure1.append(pressure)
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for i in range(len(x1)):
    x=x1[i]
    y=y1[i]
    z=z1[i]
    density=density1[i]
    velocity_x=velocity_x1[i]
    velocity_y=velocity_y1[i]
    velocity_z=velocity_z1[i]
    energy=energy1[i]
    pressure=pressure1[i]

        
    axs[0].scatter(x, density, s=1,label='With viscosity' if file == visc else 'Without viscosity')
    axs[0].set_title('Density')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Density (Kg/m^3)')
    axs[0].set_xlim([-0.6, 0.6])
    axs[0].set_ylim([np.min(density)-0.1, np.max(density)+0.1])

    axs[1].scatter(x, pressure, s=1,label='With viscosity' if file == visc else 'Without viscosity')
    axs[1].set_title('Pressure')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Pressure (N/m^3)')
    axs[1].set_xlim([-0.6, 0.6])
    axs[1].set_ylim([np.min(pressure)-0.1, np.max(pressure)+0.1])

    axs[2].scatter(x, velocity_x, s=1,label='With viscosity' if file == visc else 'Without viscosity')
    axs[2].set_title('velocity(x)')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Velocity(x) (m/s)')
    axs[2].set_xlim([-0.6, 0.6])
    axs[2].set_ylim([np.min(velocity_x)-0.1, np.max(velocity_x)+0.1])


    axs[3].scatter(x, energy, s=1,label='With viscosity' if file == visc else 'Without viscosity')
    axs[3].set_title('Internal energy')
    axs[3].set_xlabel('x')
    axs[3].set_ylabel('Internal energy (J/Kg)')
    axs[3].set_xlim([-0.6, 0.6])
    axs[3].set_ylim([np.min(energy)-0.1, np.max(energy)+0.1])

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()

#save figure
plt.savefig('1D_SPH_simulation.png', dpi=300)
plt.close()



#plot them separately
fig=plt.figure()
for i in range(len(x1)):
    x=x1[i]
    y=y1[i]
    z=z1[i]
    density=density1[i]
    velocity_x=velocity_x1[i]
    velocity_y=velocity_y1[i]
    velocity_z=velocity_z1[i]
    energy=energy1[i]
    pressure=pressure1[i]

    plt.scatter(x, density, s=1,label='With viscosity' if i == 0 else 'Without viscosity')

plt.title('Density')
plt.xlabel('x')
plt.ylabel('Density (Kg/m^3)')
plt.xlim([-0.6, 0.6])
plt.legend()
plt.savefig('Density.png', dpi=300)


fig=plt.figure()
for i in range(len(x1)):
    x=x1[i]
    y=y1[i]
    z=z1[i]
    density=density1[i]
    velocity_x=velocity_x1[i]
    velocity_y=velocity_y1[i]
    velocity_z=velocity_z1[i]
    energy=energy1[i]
    pressure=pressure1[i]

    plt.scatter(x, pressure, s=1,label='With viscosity' if i == 0 else 'Without viscosity')

plt.title('Pressure')
plt.xlabel('x')
plt.ylabel('Pressure (N/m^3)')
plt.xlim([-0.6, 0.6])
plt.legend()
plt.savefig('Pressure.png', dpi=300)

fig=plt.figure()
for i in range(len(x1)):
    x=x1[i]
    y=y1[i]
    z=z1[i]
    density=density1[i]
    velocity_x=velocity_x1[i]
    velocity_y=velocity_y1[i]
    velocity_z=velocity_z1[i]
    energy=energy1[i]
    pressure=pressure1[i]

    plt.scatter(x, velocity_x, s=1,label='With viscosity' if i == 0 else 'Without viscosity')

plt.title('velocity(x)')
plt.xlabel('x')
plt.ylabel('Velocity(x) (m/s)')
plt.xlim([-0.6, 0.6])
plt.legend()
plt.savefig('velocity_x.png', dpi=300)

fig=plt.figure()
for i in range(len(x1)):
    x=x1[i]
    y=y1[i]
    z=z1[i]
    density=density1[i]
    velocity_x=velocity_x1[i]
    velocity_y=velocity_y1[i]
    velocity_z=velocity_z1[i]
    energy=energy1[i]
    pressure=pressure1[i]

    plt.scatter(x, energy, s=1,label='With viscosity' if i == 0 else 'Without viscosity')

plt.title('Internal energy')
plt.xlabel('x')
plt.ylabel('Internal energy (J/Kg)')
plt.xlim([-0.6, 0.6])
plt.legend()
plt.savefig('Internal energy.png', dpi=300)

