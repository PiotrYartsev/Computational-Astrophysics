import tqdm as tqdm
import math as math
import matplotlib.pyplot as plt
import numpy as np

import jplephem as jpl 
from jplephem.spk import SPK
#get gravitational constant
G=4*math.pi**2

h=0.01
t=0
t_end=300

# get the ephemeris file today
eph = SPK.open('Project_1/de102.bsp')


#get the position and velocity of the planets


names=["Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto","Sun"]
positions=[1,2,3,4,5,6,7,8,9,10]
colors=["red","orange","blue","grey","green","purple","pink","brown","black","yellow"]
masses = [3.3011e23,4.8675e24,5.97237e24,6.4171e23,1.8986e27,5.6834e26,8.6810e25,1.0243e26,1.309e22,1.989e30]
masses=[x/masses[-1] for x in masses]
x = []
y = []
z = []
v_x = []
v_y = []
v_z = []

for name, i in zip(names, positions):
    # get the position and velocity of the planet
    pos, vel = eph[0,i].compute_and_differentiate(2451545.0)
    x.append(pos[0])
    y.append(pos[1])
    z.append(pos[2])
    v_x.append(vel[0])
    v_y.append(vel[1])
    v_z.append(vel[2])
    # get the mass of the planet
        

#Convert to AU and AU/year

x = np.array(x)/1.496e8
y= np.array(y)/1.496e8

z = np.array(z)/1.496e8
v_x = np.array(v_x)/1.496e8*365.25
v_y = np.array(v_y)/1.496e8*365.25
v_z = np.array(v_z)/1.496e8*365.25

x=x-x[9]
y=y-y[9]
z=z-z[9]

v_x=v_x-v_x[9]
v_y=v_y-v_y[9]
v_z=v_z-v_z[9]



W=np.zeros((len(x),6))


for i in range(len(x)):
    W[i]=np.array([x[i],y[i],z[i],v_x[i],v_y[i],v_z[i]])


print(W)
energy_list = []


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


scatters = [ax.scatter([result[0, 1, 0]], [result[0, 1, 1]], [result[0, 1, 2]], s=20, facecolor="black",edgecolor='none')]



# only keep every 10th point to make the animation faster
result = result[::10]

# Function to update the animation at each time step



# in reulst find the largest x y and z values and use them to set the limits of the plot

#print hte maximum x,yand z value for each planet

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




#set their color and size

def update(num):
    for text in ax.texts:
            text.remove()
    

    for i, scatter in enumerate(scatters):
        #If point is outside the limits of the plot, remove it
        if result[num, i, 0] > x_max or result[num, i, 0] < x_min or result[num, i, 1] > y_max or result[num, i, 1] < y_min or result[num, i, 2] > z_max or result[num, i, 2] < z_min:
            pass
        else:
            
            #scatter.set_data(result[num, i, 0], result[num, i, 1])
            scatter.set_offsets(result[num,:,:2])
            #scatter.set_color(colors[i])
            scatter.set_3d_properties(result[num, i, 2],zdir='z')
           


        


        # Adding the labels for the planet
        
        #ax.text(result[num, i, 0], result[num, i, 1], result[num, i, 2], names[i], color='black')
    ax.set_title('Solar system {} years'.format(round(num*h*300/365,1)))

#set the color of the planets in the plot

#ax.legend(names,loc='upper left',bbox_to_anchor=(1,1),prop={'size': 10})
    
    
        


# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, len(result)), interval=10, blit=False)


# Save the animation
writer=animation.FFMpegWriter(fps=30,extra_args=['-vcodec', 'libx264'])
ani.save('whole_solar_system.mp4',writer=writer)





import time 

time.sleep(1)
print("5")
time.sleep(1)
print("4")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")

#plt.show()
plt.close()

#set their color and size

x_max=40
x_min=-40
y_max=40
y_min=-40
z_max=16
z_min=-16
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

            
        #scatter.set_data(result[num, i, 0], result[num, i, 1])
        scatter.set_offsets(result[num,:,:2])
        #scatter.set_color(colors[i])
        scatter.set_3d_properties(result[num, i, 2],zdir='z')
           


        


        # Adding the labels for the planet
        
        #ax.text(result[num, i, 0], result[num, i, 1], result[num, i, 2], names[i], color='black')
    ax.set_title('Solar system {} years'.format(round(num*h*300/365,1)))

#set the color of the planets in the plot

#ax.legend(names,loc='upper left',bbox_to_anchor=(1,1),prop={'size': 10})
    
    
        


# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, len(result)), interval=10, blit=False)


# Save the animation
writer=animation.FFMpegWriter(fps=30,extra_args=['-vcodec', 'libx264'])
#ani.save('whole_solar_system_zoom.mp4',writer=writer)

plt.close()



# Display the animation





fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(x)):
    if result[:,i,0].max() > 500 or result[:,i,0].min() < -500 or result[:,i,1].max() > 500 or result[:,i,1].min() < -500 or result[:,i,2].max() > 500 or result[:,i,2].min() < -500:
        pass
    else:
        #make the edgecolor transparent so that the color of the planet is the same as the color of the path with thin lines
        ax.scatter(result[:,i,0],result[:,i,1],result[:,i,2],alpha=0.5,facecolor=colors[i],edgecolor='none',label=names[i],s=2)
        #set their color and size


#add legend for the planets and their color
ax.legend(names,loc='upper left',bbox_to_anchor=(1,1),prop={'size': 10})
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Path of planets in the solar system')

#make the plot fullscreen before saving it
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

#make the figure tight so that the legend is not cut off
plt.tight_layout()
plt.savefig('whole_solar_system.png')
plt.show()
plt.close()


#same plot, but zoomed in the mercurian orbit

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(x)):
    if result[:,i,0].max() > 20 or result[:,i,0].min() < -20 or result[:,i,1].max() > 20 or result[:,i,1].min() < -20 or result[:,i,2].max() > 20 or result[:,i,2].min() < -20:
        pass
    else:
        #make the edgecolor transparent so that the color of the planet is the same as the color of the path with thin lines
        ax.scatter(result[:,i,0],result[:,i,1],result[:,i,2],alpha=0.5,facecolor=colors[i],edgecolor='none',label=names[i],s=2)
        #set their color and size


#add legend for the planets and their color
ax.legend(names,loc='upper left',bbox_to_anchor=(1,1),prop={'size': 10})
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Path of planets in the solar system')
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.tight_layout()
plt.savefig('whole_solar_system_zoomedin.png')
plt.show()








#import linspace to make a list of time values
from numpy import linspace
time=linspace(0,300,len(energy_list))
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.plot(time,energy_list)
plt.xlabel('time [years]')
plt.ylabel('total energy [J]')
plt.title('Total energy of the solar system')
plt.savefig('total_energy_whole_system.png')
plt.show()









#"""