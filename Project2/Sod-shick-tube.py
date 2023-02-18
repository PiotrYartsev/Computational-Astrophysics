from tqdm.notebook import tqdm
import math as math
import matplotlib.pyplot as plt
import scipy as scipy
import numpy as np
from numpy.linalg import norm 
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
#set the initial conditions

# density, velocity, pressure, eneergy, distance between particles
initial_conditions_x_less_or_equal_0=[1,0,0,0,2.5]
initial_conditions_x_greater_0=[0.25,0,0,0,1.795]

mass_of_particle=0.001875
visc=0

#Populate the x axis
less=0.0075
more=0.001875
x_list=[]
start=-0.6
for i in range(320):
    x_list.append(start)
    start=start+more
for i in range(80):
    x_list.append(start)
    start=start+less



#order the x_list
#x_list=np.sort(x_list)

#create an empty state vector
State_vector=np.zeros((len(x_list),len(initial_conditions_x_less_or_equal_0)+3))



#populate the state vector
for i in range(len(x_list)):
    if x_list[i]<=0:
        State_vector[i]=[x_list[i]]+[0]+[0]+initial_conditions_x_less_or_equal_0
    else:
        State_vector[i]=[x_list[i]]+[0]+[0]+initial_conditions_x_greater_0


#position in x direction 0
x=State_vector[:,0]

#d=1
#h_1=1.3*(mass_of_particle/initial_conditions_x_less_or_equal_0[0])**(1/d)
#h_2=1.3*(mass_of_particle/initial_conditions_x_greater_0[0])**(1/d)






#kernel functions

def W(dx,h):
    a_d=1/h
    r=norm(dx)
    R=r/h
    if R<=1 and R>=0:
        return a_d * (2/3 - R**2 + 1/2 * R**3)
    else:
        return a_d * (1/6 * (2-R)**3)


#derivative of the kernel function
def W_derivat(dx,h):
    a_d=1/h
    r=norm(dx)
    R=r/h
    if R<=1 and R>=0:
        return a_d * (-2 + 3/2 * R) * dx / h**2
    else:
        return -a_d * ((1/2) * (2 - R)**2) * dx / (h * r)

#position in x direction 0
#position in y direction 1
#position in z direction 2
#density 3 
#velocity in x direction 4
#velocity in y direction 5
#velocity in z direction 6
#energy 7
import time as time

def G_function(t, State_vector):
    State_vector = State_vector.reshape((len(x_list), len(initial_conditions_x_less_or_equal_0) + 3))
    #print(State_vector[0])
    State_vector_dir = np.zeros((len(x_list), len(initial_conditions_x_less_or_equal_0) + 3))


    #just use a defoult value
    h_1=0.01878527*5
    h_2=0.01878527
    
    h_list=[]
    for i in range(320):
        h_list.append(h_1)
    for i in range(320,len(x_list)):
        h_list.append(h_2)

    k=2
    dx = State_vector[:, 0] - State_vector[:, 0][:, np.newaxis]

    W_value = np.zeros((len(x), len(x)))
    Delta_W_value = np.zeros((len(x), len(x)))
    h_tes=[]

    for i in range(len(x)):
        for j in range(len(x)):
            h=(h_list[i]+h_list[j])/2
            if h not in h_tes:
                h_tes.append(h)
            if norm(dx[i, j])<=(k*(h)):
                W_value[i, j] = W(dx[i,j], h)
                #print(W_value[i, j])
                Delta_W_value[i, j] = W_derivat(dx[i, j], h)
                #print(Delta_W_value[i, j])
    
    

    State_vector_dir[:, 0] = State_vector[:, 4]
    State_vector_dir[:, 1] = State_vector[:, 5]
    State_vector_dir[:, 2] = State_vector[:, 6]

    gamma = 1.4
    pressure = (gamma - 1) * State_vector[:, 3] * State_vector[:, 7]
    seed_of_sound = np.sqrt((gamma - 1) * State_vector[:, 7])

    temp1 = pressure / (State_vector[:, 3]**2) + pressure[:, np.newaxis] / (State_vector[:, 3][:, np.newaxis]**2) + visc
    temp2 = (State_vector[:, 4] - State_vector[:, 4][:, np.newaxis]) * Delta_W_value

    State_vector_dir[:, 7] = 1/2 * np.sum(mass_of_particle * (temp1 * temp2), axis=1)
    State_vector_dir[:, 4] = -np.sum(mass_of_particle * (temp1 * Delta_W_value), axis=1)
    #State_vector_dir[:, 3] = np.sum(mass_of_particle * ((State_vector[:, 4] - State_vector[:, 4][:, np.newaxis]) * Delta_W_value), axis=1)
    State_vector_dir[:,3]=0
    State_vector[:,3]=np.sum(mass_of_particle * W_value, axis=1)

    #plt.plot(State_vector[:,0],pressure)
    #plt.show(block=False)
    #plt.pause(2)
    #plt.show()
    #plt.close()

    
    return State_vector_dir.reshape(-1)


#import a function to calculate the Runge-Kutta method
from scipy.integrate import RK45 as RK45

#flatten the state vector
result = []
result.append(State_vector)
State_vector=State_vector.reshape(-1)
#run it once to see if it works
G_function(0,State_vector)


#add the strating state to the result

# Set the initial conditions

t=0
t_end=0.1
h=0.005
# Initialize the RK45 integrator
integrator = RK45(G_function, t, State_vector, t_end, h)

# Integrate the equations of motion
# set the total progress to be t_end/h
#prevent the progress bar to be reprinted and update the progress bar
from tqdm.auto import tqdm
with tqdm(total=int(t_end/h)) as pbar:
    while integrator.t < t_end:
        integrator.step()
        pbar.update()
        final_state = integrator.y
        final_state = final_state.reshape((len(x), len(initial_conditions_x_less_or_equal_0) + 3))
        result.append(final_state)

result=np.stack(result)

#for each state of the system, reshape the array to the original shape 

section=result[-1]
x=section[:,0]
density=section[:,3]

energy=section[:,7]
gamma=1.4
pressure=(gamma-1)*density*energy

velocity_x=section[:,4]
x, density, pressure, velocity_x, energy = zip(*sorted(zip(x, density, pressure, velocity_x, energy)))

#make 4 plots side by side
fig, axs = plt.subplots(1, 4,figsize=(20,5))
axs[0].plot(x,density)
axs[0].set_title('density')
axs[1].plot(x,pressure)
axs[1].set_title('pressure')
axs[2].plot(x,velocity_x)
axs[2].set_title('velocity_x')
axs[3].plot(x,energy)
axs[3].set_title('energy')
plt.show()

for section in result:
    x=section[:,0]
    
    density=section[:,3]

    energy=section[:,7]
    gamma=1.4
    pressure=(gamma-1)*density*energy

    velocity_x=section[:,4]

    #order all list by x
    x, density, pressure, velocity_x, energy = zip(*sorted(zip(x, density, pressure, velocity_x, energy)))
    #make 4 plots side by side
    fig, axs = plt.subplots(1, 4,figsize=(20,5))
    axs[0].plot(x,density)
    axs[0].set_title('density')
    axs[1].plot(x,pressure)
    axs[1].set_title('pressure')
    axs[2].plot(x,velocity_x)
    axs[2].set_title('velocity_x')
    axs[3].plot(x,energy)
    axs[3].set_title('energy')
    plt.show()
#"""