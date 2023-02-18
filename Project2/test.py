from tqdm.notebook import tqdm
import math as math
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm 
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

# set the initial conditions
# density, velocity, pressure, energy, distance between particles
initial_conditions_x_less_or_equal_0 = [1, 0, 0, 0, 2.5]
initial_conditions_x_greater_0 = [0.25, 0, 0, 0, 1.795]

mass_of_particle = 0.001875
visc = 0

# Populate the x axis
small_step = 0.6 / 80
start = -0.6
State_vector = []
for i in np.linspace(-0.6, 0, 320):
    State_vector.append((i, 0, 0, initial_conditions_x_less_or_equal_0[0], initial_conditions_x_less_or_equal_0[1], initial_conditions_x_less_or_equal_0[2], initial_conditions_x_less_or_equal_0[3], initial_conditions_x_less_or_equal_0[4]))
for i in np.linspace(0 + small_step, 0.6 + small_step, 80):
    State_vector.append((i, 0, 0, initial_conditions_x_greater_0[0], initial_conditions_x_greater_0[1], initial_conditions_x_greater_0[2], initial_conditions_x_greater_0[3], initial_conditions_x_greater_0[4]))
State_vector = np.array(State_vector)
x = State_vector[:, 0]
print(State_vector.shape)
print(State_vector[0])

# d = 1
# h_1 = 1.3 * (mass_of_particle / initial_conditions_x_less_or_equal_0[0])**(1 / d)
# h_2 = 1.3 * (mass_of_particle / initial_conditions_x_greater_0[0])**(1 / d)

# kernel functions
def W(dx, h):
    a_d = 1 / h
    r = norm(dx)
    R = r / h
    if R <= 1 and R >= 0:
        return a_d * (2 / 3 - R**2 + 1 / 2 * R**3)
    else:
        return a_d * (1 / 6 * (2 - R)**3)


# derivative of the kernel function
def W_derivat(dx, h):
    a_d = 1 / h
    r = norm(dx)
    R = r / h
    if R <= 1 and R >= 0:
        return a_d * (-2 + 3 / 2 * R) * dx / h**2
    else:
        return -a_d * ((1 / 2) * (2 - R)**2) * dx / (h * r)


# position in x direction 0
# position in y direction 1
# position in z direction 2
# density 3 
# velocity in x direction 4
# velocity in y direction 5
# velocity in z direction 6
# energy 7
import time as time


def G_function(State_vector, t):
    State_vector_dir = np.zeros((400, len(initial_conditions_x_less_or_equal_0) + 3))

    # just use a default value
    h_1 = 0.01878527 * 5
    h_2 = 0.01878527

    h_list = []
    for i in range(320):
        h_list.append(h_1)
    for i in range(320, 400):
        h_list.append(h_2)

    k = 2
    dx = State_vector[:, 0] - State_vector[:, 0][:, np.newaxis]

    W_value = np.zeros((400, 400))
    Delta_W_value = np.zeros((400, 400))
    h_tes = []

    for i in range(400):
        for j in range(400):
            h = (h_list[i] + h_list[j]) / 2
            if h not in h_tes:
                h_tes.append(h)
            if norm(dx[i, j]) <= (k * (h)):
                W_value[i, j] = W(dx[i, j], h)
                Delta_W_value[i, j] = W_derivat(dx[i, j], h)

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
    State_vector_dir[:, 3] = 0
    State_vector[:, 3] = np.sum(mass_of_particle * W_value, axis=1)
    State_vector_dir[:, 5:8] = 0

    for i in range(400):
        for j in range(400):
            if i != j and norm(dx[i, j]) <= (k * (h_list[i] + h_list[j]) / 2):
                State_vector_dir[i, 5:8] += mass_of_particle * State_vector[j, 3] * (State_vector[j, 5:8] - State_vector[i, 5:8]) * Delta_W_value[i, j]

    return State_vector_dir.flatten()


#numerically solve the ODE
from scipy.integrate import solve_ivp
#initial time
t0=0
#final time
tmax=0.1

State_vector_reshaped=State_vector.ravel()
solution=solve_ivp(G_function, (t0,tmax), State_vector_reshaped)

#split the solution vector back into a 2d array
solution_array=np.reshape(solution.y,(400,8,int(solution.t.size)))

#print(np.array(solution_array).shape)
#print(np.array(solution_array)[0])
fig, axs = plt.subplots(1, 1, figsize=(8, 8), dpi=80)
axs.scatter(solution_array[:,0, -1], solution_array[:, 4, -1], alpha=0.5)
axs.set_xlim([-0.6, 0.6])
axs.set_ylim([-0.5, 0.5])
plt.show()