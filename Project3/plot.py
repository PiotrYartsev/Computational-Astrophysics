
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import RK45
import matplotlib.animation as animation
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
where='C:\\Users\\piotr\\Documents\\GitHub\\Computational-Astrophysics\\300planet-diffmass-headon1.npy'
S_i_300 = np.load(where)

if "1" in where:
    planets=2
    print("2 planet")
else:
    print("1 planets")
    planets=1
print(S_i_300.shape)


x= S_i_300[:,:,0]
y= S_i_300[:,:,1]
z= S_i_300[:,:,2]
vx= S_i_300[:,:,3]
vy= S_i_300[:,:,4]
vz= S_i_300[:,:,5]
densities= S_i_300[:,:,6]
pressure= S_i_300[:,:,7]
energy= S_i_300[:,:,8]

h=1000/3600 
t=0
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
bar=tqdm(total=len(x))
def animate(i):
    ax.clear()
    bar.update(1)
    ax.set_xlim3d(np.min(x), np.max(x))
    ax.set_ylim3d(np.min(y), np.max(y))
    ax.set_zlim3d(np.min(z), np.max(z))
    ax.view_init(elev=40, azim=30) 
    ax.scatter(x[i], y[i], z[i], c='r', marker='o', s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Simulation at t = {:.2f} in days'.format(t + i * h))
    # adjust the viewing angle so that we are looking down on the xy plane with a about a 30 degree angle
    ax.view_init(elev=40, azim=30)

#anim = FuncAnimation(fig, update, frames=result.shape[0], init_func=init, blit=True, interval=100)
anim = FuncAnimation(fig, animate, frames=len(x), interval=100, repeat=False)
writer=animation.FFMpegWriter(fps=50,extra_args=['-vcodec', 'libx264'])
anim.save('{}_planet.mp4'.format(planets),writer=writer)
#plt.show()
print('\n done {}_planet.mp4'.format(planets))


plt.close()


x= S_i_300[-1,:,0]
y= S_i_300[-1,:,1]
z= S_i_300[-1,:,2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(np.min(x), np.max(x))
ax.set_ylim3d(np.min(y), np.max(y))
ax.set_zlim3d(np.min(z), np.max(z))
ax.view_init(elev=40, azim=30) 
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Final state of the simulation')
plt.show()
plt.savefig('final_state_{}.png'.format(planets))
plt.close()

#make 6 figures in asingle plot

#find where in the x vector we hav gone 6 equal parts
#then plot the 6 figures in a single plot
"""
#positions = np.linspace(0, len(x)-1, 6, dtype=int)
positions = np.unique((np.linspace(0, len(x) ** 1.25 - 1, 6, dtype=int) ** (1 / 1.5)).astype(int))
print(len(positions))

x1= S_i_300[positions[0],:,0]
y1= S_i_300[positions[0],:,1]

dems1= densities[positions[0],:]

x2= S_i_300[positions[1],:,0]
y2= S_i_300[positions[1],:,1]
dems2= densities[positions[1],:]

x3= S_i_300[positions[2],:,0]
y3= S_i_300[positions[2],:,1]
dems3= densities[positions[2],:]

x4= S_i_300[positions[3],:,0]
y4= S_i_300[positions[3],:,1]
dems4= densities[positions[3],:]

x5= S_i_300[positions[4],:,0]
y5= S_i_300[positions[4],:,1]
dems5= densities[positions[4],:]

x6= S_i_300[positions[5],:,0]
y6= S_i_300[positions[5],:,1]
dems6= densities[positions[5],:]

#make a 2d plot of the x and y positions of the particles and color them by density with a gradient bar on the side
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='col', sharey='row')
ax1.scatter(x1, y1, c=dems1, marker='o', s=5)
ax1.set_title('t = {:.2f} days'.format(t + positions[0] * h))
ax2.scatter(x2, y2, c=dems2, marker='o', s=5)
ax2.set_title('t = {:.2f} days'.format(t + positions[1] * h))
ax3.scatter(x3, y3, c=dems3, marker='o', s=5)
ax3.set_title('t = {:.2f} days'.format(t + positions[2] * h))
ax4.scatter(x4, y4, c=dems4, marker='o', s=5)
ax4.set_title('t = {:.2f} days'.format(t + positions[3] * h))
ax5.scatter(x5, y5, c=dems5, marker='o', s=5)
ax5.set_title('t = {:.2f} days'.format(t + positions[4] * h))
ax6.scatter(x6, y6, c=dems6, marker='o', s=5)
ax6.set_title('t = {:.2f} days'.format(t + positions[5] * h))
fig.suptitle('Density of the {} planet system'.format(planets))
#leave some space between the plots
fig.subplots_adjust(hspace=0.5)


plt.savefig('density_{}.png'.format(planets))
plt.close()


#positions = np.linspace(0, len(x)-1, 6, dtype=int)
positions = np.unique((np.linspace(0, len(x) ** 1.25 - 1, 6, dtype=int) ** (1 / 1.5)).astype(int))
print(len(positions))

x1= S_i_300[positions[0],:,0]
y1= S_i_300[positions[0],:,1]

velocity1= np.sqrt(S_i_300[positions[0],:,3]**2 + S_i_300[positions[0],:,4]**2 + S_i_300[positions[0],:,5]**2)

x2= S_i_300[positions[1],:,0]
y2= S_i_300[positions[1],:,1]
velocity2= np.sqrt(S_i_300[positions[1],:,3]**2 + S_i_300[positions[1],:,4]**2 + S_i_300[positions[1],:,5]**2)

x3= S_i_300[positions[2],:,0]
y3= S_i_300[positions[2],:,1]
velocity3= np.sqrt(S_i_300[positions[2],:,3]**2 + S_i_300[positions[2],:,4]**2 + S_i_300[positions[2],:,5]**2)

x4= S_i_300[positions[3],:,0]
y4= S_i_300[positions[3],:,1]
velocity4= np.sqrt(S_i_300[positions[3],:,3]**2 + S_i_300[positions[3],:,4]**2 + S_i_300[positions[3],:,5]**2)

x5= S_i_300[positions[4],:,0]
y5= S_i_300[positions[4],:,1]
velocity5= np.sqrt(S_i_300[positions[4],:,3]**2 + S_i_300[positions[4],:,4]**2 + S_i_300[positions[4],:,5]**2)

x6= S_i_300[positions[5],:,0]
y6= S_i_300[positions[5],:,1]
velocity6= np.sqrt(S_i_300[positions[5],:,3]**2 + S_i_300[positions[5],:,4]**2 + S_i_300[positions[5],:,5]**2)

#make a 2d plot of the x and y positions of the particles and color them by density with a gradient bar on the side
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='col', sharey='row')
ax1.scatter(x1, y1, c=velocity1, marker='o', s=5)
ax1.set_title('t = {:.2f} days'.format(t + positions[0] * h))
ax2.scatter(x2, y2, c=velocity2, marker='o', s=5)
ax2.set_title('t = {:.2f} days'.format(t + positions[1] * h))
ax3.scatter(x3, y3, c=velocity3, marker='o', s=5)
ax3.set_title('t = {:.2f} days'.format(t + positions[2] * h))
ax4.scatter(x4, y4, c=velocity4, marker='o', s=5)
ax4.set_title('t = {:.2f} days'.format(t + positions[3] * h))
ax5.scatter(x5, y5, c=velocity5, marker='o', s=5)
ax5.set_title('t = {:.2f} days'.format(t + positions[4] * h))
ax6.scatter(x6, y6, c=velocity6, marker='o', s=5)
ax6.set_title('t = {:.2f} days'.format(t + positions[5] * h))

fig.suptitle('Velocity of the {} planet system'.format(planets))
#leave some space between the plots
fig.subplots_adjust(hspace=0.5)


plt.savefig('velocity_{}.png'.format(planets))
plt.close()


#positions = np.linspace(0, len(x)-1, 6, dtype=int)
positions = np.unique((np.linspace(0, len(x) ** 1.25 - 1, 6, dtype=int) ** (1 / 1.5)).astype(int))
print(len(positions))

x1= S_i_300[positions[0],:,0]
y1= S_i_300[positions[0],:,1]
energy1=energy[positions[0],:]

x2= S_i_300[positions[1],:,0]
y2= S_i_300[positions[1],:,1]
energy2=energy[positions[1],:]

x3= S_i_300[positions[2],:,0]
y3= S_i_300[positions[2],:,1]
energy3=energy[positions[2],:]

x4= S_i_300[positions[3],:,0]
y4= S_i_300[positions[3],:,1]
energy4=energy[positions[3],:]

x5= S_i_300[positions[4],:,0]
y5= S_i_300[positions[4],:,1]
energy5=energy[positions[4],:]

x6= S_i_300[positions[5],:,0]
y6= S_i_300[positions[5],:,1]
energy6=energy[positions[5],:]

#make a 2d plot of the x and y positions of the particles and color them by density with a gradient bar on the side
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='col', sharey='row')
ax1.scatter(x1, y1, c=energy1, marker='o', s=5)
ax1.set_title('t = {:.2f} days'.format(t + positions[0] * h))
ax2.scatter(x2, y2, c=energy2, marker='o', s=5)
ax2.set_title('t = {:.2f} days'.format(t + positions[1] * h))
ax3.scatter(x3, y3, c=energy3, marker='o', s=5)
ax3.set_title('t = {:.2f} days'.format(t + positions[2] * h))
ax4.scatter(x4, y4, c=energy4, marker='o', s=5)
ax4.set_title('t = {:.2f} days'.format(t + positions[3] * h))
ax5.scatter(x5, y5, c=energy5, marker='o', s=5)
ax5.set_title('t = {:.2f} days'.format(t + positions[4] * h))
ax6.scatter(x6, y6, c=energy6, marker='o', s=5)
ax6.set_title('t = {:.2f} days'.format(t + positions[5] * h))

fig.suptitle('Energy of the {} planet system'.format(planets))
#leave some space between the plots
fig.subplots_adjust(hspace=0.5)


plt.savefig('energy_{}.png'.format(planets))
plt.close()

