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


print(len(W))
print(len(W[0]))






def Der_W(W,t):

    #make a 4x6 matrix of zeros
    W_derivat=np.zeros((len(W),len(W[0])))
    for i in range(3):
        W_derivat[:,i]=W[:,i+3]

    #for x, y and z
    for ii in [0,1,2]:
        r=W[:,ii]
        for position in range(len(r)):
            len_vector_r=math.sqrt((r[position]-r[0])**2)
            W_derivat[:,ii+3]=-G*masses[position]*r[position]/len_vector_r**3
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




import matplotlib.animation as animation

# Set up the figure and axes for the animation
fig, ax = plt.subplots()


# Initialize the scatter plot objects for the bodies and add labels

#rewrite results to only keep every 10th value
result=result[::10]



scatters = [ax.scatter([result[0, i, 0]], [result[0, i, 1]], s=100, label='{}'.format(names[i])) for i in range(4)]




# Function to initialize the animation
def init():
    return scatters

# Function to update the animation at each time step 
# make sure it fits the entire animation

def animate(i):
    ax.set_title('Time: {:.2f} years'.format(i*h))
    #add a legend
    ax.legend()
    

    #add x and y labels
    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')

    #set the limits of the plot that it fits the entire animation 
    

    largest_x=-10000
    smallest_x=10000
    largest_y=-10000
    smallest_y=10000


    for j, scatter in enumerate(scatters):
        scatter.set_offsets([result[i, j, 0], result[i, j, 1]])
        if result[i, j, 0]>largest_x:
            largest_x=result[i, j, 0]
        if result[i, j, 0]<smallest_x:
            smallest_x=result[i, j, 0]
        if result[i, j, 1]>largest_y:
            largest_y=result[i, j, 1]
        if result[i, j, 1]<smallest_y:
            smallest_y=result[i, j, 1]
    if largest_x>-smallest_x:
        ax.set_xlim(-largest_x-5, largest_x+5)
    else:
        ax.set_xlim(smallest_x-5, -smallest_x+5)
    if largest_y>-smallest_y:
        ax.set_ylim(-largest_y-5, largest_y+5)
    else:
        ax.set_ylim(smallest_y-5, -smallest_y+5)
    return scatters



# Create the animation with PillowWriter
ani = animation.FuncAnimation(fig, animate, frames=result.shape[0], init_func=init, blit=True, interval=1)



# Save the animation
from matplotlib.animation import PillowWriter
ani.save('Computational-Astrophysics/Project_1/Starting_values.gif', writer=PillowWriter(fps=30))



# Display the animation

#plt.show()
print("Done")