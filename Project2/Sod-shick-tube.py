from tqdm.notebook import tqdm
import math as math
import matplotlib.pyplot as plt
import scipy as scipy
import numpy as np


#set the initial conditions

# density, velocity, pressure, eneergy, distance between particles
initial_conditions_x_less_or_equal_0=[1,0,0,0,2.5]
initial_conditions_x_greater_0=[0.25,0,0,0,1.795]

mass_of_particle=0.001875
visc=0

#Populate the x axis
x_less_than_o=np.linspace(0+0.0075,6,320)
x_greater_than_0=np.linspace(0,-6+0.001875,80)
x_list=np.concatenate((x_less_than_o,x_greater_than_0),axis=0)
#order the x_list
x_list=np.sort(x_list)

#create an empty state vector
State_vector=np.zeros((len(x_list),len(initial_conditions_x_less_or_equal_0)+3))



#populate the state vector
for i in range(len(x_list)):
    if x_list[i]<=0:
        State_vector[i]=[x_list[i]]+[0]+[0]+initial_conditions_x_less_or_equal_0
    else:
        State_vector[i]=[x_list[i]]+[0]+[0]+initial_conditions_x_greater_0


#position in x direction 0
x=np.concatenate((x_less_than_o,x_greater_than_0),axis=0)
#order 

"""
"""
#define the smoothing length
d=1
h_1=1.3*(initial_conditions_x_less_or_equal_0[-2]/initial_conditions_x_less_or_equal_0[4])**(1/d)
h_2=1.3*(initial_conditions_x_greater_0[-2]/initial_conditions_x_greater_0[4])**(1/d)
h=(h_1+h_2)/2
"""
"""
#just use a defoult value
h=0.001875*20

a_d=1/h



#kernel functions

def W(R, r, a_d, h):
    output = np.zeros_like(R)
    mask1 = (R >= 0) & (R <= 1)
    mask2 = (R > 1) & (R <= 2)
    output[mask1] = a_d * (2/3 - R[mask1]**2 + 1/2 * R[mask1]**3)
    output[mask2] = a_d * (1/6 * (2-R[mask2])**3)
    return output




#derivative of the kernel function

def W_derivat(R, r, a_d, h, dx):
    mask1 = (R <= 1) & (R >= 0) | (R == 0)
    mask2 = (R > 1) & (R <= 2)
    
    output = np.zeros_like(R)
    output[mask1] = a_d * (-2 + 3/2 * R[mask1]) * dx[mask1] / h**2
    output[mask2] = a_d * (-(1/2) * (2 - R[mask2])**2) * dx[mask2] / (h * r[mask2])
    
    return output

def density_function(mass,velocity_i, velocity_j, delta_W_ij):
    return mass*(velocity_i-velocity_j)*delta_W_ij

def velocity_function(mass,velocity_i, velocity_j,pressure_i,pressure_j,density_i, density_j, artvisc, delta_W_ij):
    return -mass*(pressure_i/(density_i**2)+pressure_j/(density_j**2)+artvisc)*delta_W_ij

def energy_function(mass,velocity_i, velocity_j,pressure_i,pressure_j,density_i, density_j, artvisc, delta_W_ij):
    return -1/2 * mass*(velocity_i*pressure_i/(density_i**2)+velocity_j*pressure_j/(density_j**2)+artvisc*velocity_i)*delta_W_ij
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

    State_vector_dir = np.zeros((len(x_list), len(initial_conditions_x_less_or_equal_0) + 3))

    r_sign = State_vector[:, 0] - State_vector[:, 0][:, np.newaxis]
    r = np.sqrt(r_sign**2)
    R = r/h

    W_value = np.zeros((len(x), len(x)))
    Delta_W_value = np.zeros((len(x), len(x)))

    W_value = W(R, r, a_d, h)
    Delta_W_value = W_derivat(R, r, a_d, h, r_sign)

    W_value[np.eye(len(x), dtype=bool)] = 0
    Delta_W_value[np.eye(len(x), dtype=bool)] = 0

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
    State_vector_dir[:, 3] = np.sum(mass_of_particle * ((State_vector[:, 4] - State_vector[:, 4][:, np.newaxis]) * Delta_W_value), axis=1)

    return State_vector_dir.reshape(-1)




#solve the differential equation using runge kutta

#import a function to calculate the Runge-Kutta method
from scipy.integrate import RK45 as RK45

#flatten the state vector


State_vector=State_vector.reshape(-1)
#run it once to see if it works
print(G_function(0,State_vector))


result = []
t=0
t_end=40
h=0.005
# Initialize the RK45 integrator
integrator = RK45(G_function, t, State_vector, t_end, h,atols=1e-10,rtols=1e-10)

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

#save result
with open('my_array.csv', 'w') as my_file:
        for i in result:
            np.savetxt(my_file,i)
print('Array exported to file')
"""
result=[]
with open('my_array.csv', 'r') as my_file:
    for line in my_file:
        result.append(line)
result=np.stack(result)


for i in range(len(result)):
    print(result[i])
    x_values=result[i,:,0]  
    density_values=result[i,:,3]
    plt.plot(x_values,density_values)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("density vs x")
    plt.show()
#"""