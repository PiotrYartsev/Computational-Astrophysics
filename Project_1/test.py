import numpy as np

# Gravitational constant in units of AU^3 / (solar mass * year^2)
G = 4 * np.pi ** 2

# Initial conditions for the Sun, Jupiter, and Trojan satellites
masses = np.array([1.0, 0.001, 0, 0])
x_init = np.array([[0.0, 0.0, 0.0], [0.0, 5.2, 0.0], [0.0, -4.503, 2.6], [0.0, 4.503, 2.6]])
v_init = np.array([[0.0, 0.0, 0.0], [-2.75674, 0.0, 0.0], [0.0, -1.38, -2.39], [0.0, -1.38, 2.39]])

# Time step for numerical integration
dt = 0.01

# Total time for the simulation
total_time = 300.0

# Number of time steps
num_steps = int(total_time / dt)

# Arrays to store the position and velocity of each body at each time step
x = np.zeros((num_steps + 1, 4, 3))
v = np.zeros((num_steps + 1, 4, 3))

# Set initial conditions
x[0] = x_init
v[0] = v_init

# 4th order Runge-Kutta numerical integrator
for i in range(num_steps):
    # Compute the forces acting on each body
    F = np.zeros((4, 3))
    for j in range(4):
        for k in range(4):
            if j != k:
                r = x[i, j] - x[i, k]
                F[j] += -G * masses[j] * masses[k] * r / np.linalg.norm(r) ** 3
    # Update the position and velocity of each body using the Runge-Kutta method
    k1v = dt * F / masses[:, None]
    k1x = dt * v[i]
    k2v = dt * F / masses[:, None]
    k2x = dt * (v[i] + 0.5 * k1v)
    k3v = dt * F / masses[:, None]
    k3x = dt * (v[i] + 0.5 * k3v)
    k4v = dt * F / masses[:, None]
    k4x = dt * (v[i] + k3v)
    v[i+1] = v[i] + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    x[i+1] = x[i] + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    # subtracting from every body the forces exerted by all other bodies on the Sun
    x[i+1][0] = 0
    v[i+1][0] = 0

import matplotlib.animation as animation
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

def init():
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    return

def animate(i):
    ax.clear()
    ax.set_title('Time: {:.2f} years'.format(i*dt))
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    for j in range(2, 4):
        ax.scatter(x[i, j, 0], x[i, j, 1], label='Trojan {}'.format(j))
    ax.scatter(x[i, 1, 0], x[i, 1, 1], label='Jupiter')
    ax.legend()
    return

ani = animation.FuncAnimation(fig, animate, frames=num_steps, init_func=init, repeat=False)
plt.show()