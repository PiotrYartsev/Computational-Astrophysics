import tqdm as tqdm
import math as math
import matplotlib.pyplot as plt

#get gravitational constant
G=4*math.pi**2

h=0.01
t=0
t_end=300

Sun_mass=1
Jupiter_mass=0.001
Trojan1_mass=0
Trojan2_mass=0

masses=[Sun_mass,Jupiter_mass,Trojan1_mass,Trojan2_mass]
names=["Sun","Jupiter","Trojan1","Trojan2"]
colors=["yellow","green","red","blue"]
x=[0,0,-4.503,4.503]
y=[0, 5.2, 2.6, 2.6]
z=[0,0,0,0]

v_x=[0, -2.75674, -1.38, -1.38]
v_y=[0,0, -2.39, 2.39]
v_z=[0,0,0,0]

#make an matrix of the starting values
import numpy as np

W=np.array([[x[0],y[0],z[0],v_x[0],v_y[0],v_z[0]],[x[1],y[1],z[1],v_x[1],v_y[1],v_z[1]],[x[2],y[2],z[2],v_x[2],v_y[2],v_z[2]],[x[3],y[3],z[3],v_x[3],v_y[3],v_z[3]]])


print(W)




energy_list=[]


def Der_W(W, t):
   #initialize the derivative matrix
   W_derivat=np.zeros((len(W),len(W[0])))
   #iterate over the planets
   W_derivat=np.zeros((len(W),len(W[0])))
   for i in range(3):
       W_derivat[:,i]=W[:,i+3]
   

   E=0
   for i in range(len(x)):
        E += masses[i]*(W[i,3]**2 + W[i,4]**2 + W[i,5]**2)/2    
        for j in range(len(x)):
            if i!=j:
               r=math.sqrt((W[i,0]-W[j,0])**2+(W[i,1]-W[j,1])**2+(W[i,2]-W[j,2])**2)
               E -= G*masses[i]*masses[j]/r
               W_derivat[i,3]+=G*masses[j]*(W[j,0]-W[i,0])/r**3
               W_derivat[i,4]+=G*masses[j]*(W[j,1]-W[i,1])/r**3
               W_derivat[i,5]+=G*masses[j]*(W[j,2]-W[i,2])/r**3
      

   energy_list.append(E)
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


scatters = [ax.scatter([result[0, i, 0]], [result[0, i, 1]], [result[0, i, 2]], s=20) for i in range(4)]



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


# Save the animation as a mp4
writer=animation.FFMpegWriter(fps=30,extra_args=['-vcodec', 'libx264'])
#ani.save('solar_system.mp4',writer=writer)



# Save the animation
#ani.save('solar_system.mp4', fps=30, extra_args=['-vcodec', 'libx264'])



# Display the animation
#plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(4):
    ax.scatter(result[:,i,0],result[:,i,1],result[:,i,2],alpha=0.5,facecolor=colors[i],edgecolor='none',label=names[i],s=2)
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Path of planets in the solar system')
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

plt.tight_layout()
plt.savefig('solar_system.png')
#plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(4):
    ax.scatter(result[:,i,0],result[:,i,1],result[:,i,2],alpha=0.5,facecolor=colors[i],edgecolor='none',label=names[i],s=2)
ax.legend()

#limit the plot to the outer ring of planets from x=2 to x=4 and y=2 to y=4
ax.set_xlim(2,4)
ax.set_ylim(2,4)
ax.set_zlim(-0.5,0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Path of planets in the solar system')
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

plt.tight_layout()

plt.savefig('solar_system_zoomedin.png')
#plt.show()
plt.close()


#print(energy_list)

#import linspace to make a list of time values
from numpy import linspace
time=linspace(0,300,len(energy_list))
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
#plot the points
plt.plot(time,energy_list,'b*')
plt.xlabel('time [years]')
plt.ylabel('total energy')
plt.title('Total energy of the solar system')
plt.savefig('total_energy.png')
plt.show()

