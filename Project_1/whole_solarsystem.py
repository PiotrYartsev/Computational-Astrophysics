import tqdm as tqdm
import math as math
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3)

from jplephem.spk import SPK

# get planetary ephemeris
eph = SPK.open('/home/piotr/Documents/GitHub/Computational-Astrophysics/Project_1/de102.bsp')

print(eph)
#get the name of the planets at position 0,4
print(eph[0,4].target)

positions = eph[0,4].compute(2457061.5)

velocity = eph[0,4].compute_and_differentiate(2457061.5)
print("\n")
print(positions)
print("\n")
print(velocity)

"""
#get gravitational constant
G=4*math.pi**2

h=0.01
t=0
t_end=300

# list to store the masses of the planets
masses = []
# list to store the names of the planets
names = []

# list to store the initial positions and velocities of the planets
x = []
y = []
z = []
v_x = []
v_y = []
v_z = []

# iterate over the planets to get the positions and velocities of the planets
for planet_name in eph.body_name_to_id:
    if planet_name == 'SUN':
        mass = 1.0
    else:
        # get the mass of the planet
        mass = eph.body_name_to_id[planet_name]['mass']
    masses.append(mass)
    # get the name of the planet
    names.append(planet_name)

    # get the initial positions and velocities of the planet at t=0
    pos, vel = eph.compute_and_differentiate(planet_name, 'SUN', t)
    x.append(pos[0])
    y.append(pos[1])
    z.append(pos[2])
    v_x.append(vel[0])
    v_y.append(vel[1])
    v_z.append(vel[2])

# make a matrix of the starting values
W=np.array([[x[0],y[0],z[0],v_x[0],v_y[0],v_z[0]],[x[1],y[1],z[1],v_x[1],v_y[1],v_z[1]],[x[2],y[2],z[2],v_x[2],v_y[2],v_z[2]], ...])


print(W)





#"""
"""
def Der_W(W, t):
    #initialize the derivative matrix
    W_derivat=np.zeros((len(W),len(W[0])))
    #iterate over the planets
    W_derivat=np.zeros((len(W),len(W[0])))
    for i in range(3):
        W_derivat[:,i]=W[:,i+3]
        
    for i in range(4):
        for j in range(4):
            if i!=j:
                r=math.sqrt((W[i,0]-W[j,0])**2+(W[i,1]-W[j,1])**2+(W[i,2]-W[j,2])**2)
                W_derivat[i,3]+=G*masses[j]*(W[j,0]-W[i,0])/r**3
                W_derivat[i,4]+=G*masses[j]*(W[j,1]-W[i,1])/r**3
                W_derivat[i,5]+=G*masses[j]*(W[j,2]-W[i,2])/r**3
        

    return W_derivat


#for making a 4th order Runge-Kutta
def RK4(W,t,h,Der_W):
    k1=h*Der_W(W,t)
    k2=h*Der_W(W+0.5*k1,t+0.5*h)
    k3=h*Der_W(W+0.5*k2,t+0.5*h)
    k4=h*Der_W(W+k3,t+h)
    W_next=W+(k1+2*k2+2*k3+k4)/6
    return W_next

# To store the results
result = np.zeros((int((t_end-t)/h)+1, len(W), len(W[0])))

# Iterate over time steps using tqdm to display a progress bar
for i, t in enumerate(tqdm.tqdm(np.arange(t, t_end, h))):
    result[i] = W
    W = RK4(W, t, h, Der_W)




import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Set up the figure and axes for the animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



#make sure the sun is in the center and all the other planets position is relative to the sun



# Initialize the scatter plot objects for the bodies and with labels


scatters = [ax.scatter([result[0, i, 0]], [result[0, i, 1]], [result[0, i, 2]], s=100) for i in range(4)]



# only keep every 10th point to make the animation faster
#result=result[::10]
# Function to update the animation at each time step



# in reulst find the largest x y and z values and use them to set the limits of the plot
x_max=np.max(result[:,:,0])
x_min=np.min(result[:,:,0])
y_max=np.max(result[:,:,1])
y_min=np.min(result[:,:,1])
z_max=np.max(result[:,:,2])
z_min=np.min(result[:,:,2])
ax.set_xlim(x_min,x_max)
ax.set_ylim(y_min,y_max)
ax.set_zlim(z_min,z_max)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
def update(num):
    for text in ax.texts:
            text.remove()
    

    for i, scatter in enumerate(scatters):
        scatter.set_offsets(result[num,:,:2])
        scatter.set_3d_properties(result[num, i, 2],zdir='z')
        


        # Adding the labels for the planet
        
        ax.text(result[num, i, 0], result[num, i, 1], result[num, i, 2], names[i], color='black')
    ax.set_title('Solar system {} years'.format(round(num*h*300/365,1)))

    
    
    
        


# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, len(result)), interval=10, blit=False)


# Save the animation as a gif
#ani.save('solar_system.gif', writer='imagemagick', fps=30)


# Save the animation
#ani.save('solar_system.mp4', fps=30, extra_args=['-vcodec', 'libx264'])



# Display the animation
plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(4):
    ax.scatter(result[:,i,0],result[:,i,1],result[:,i,2],alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Path of planets in the solar system')


plt.savefig('solar_system.png')
plt.show()
#"""