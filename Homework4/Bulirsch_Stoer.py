#change the values of a and b to get different results

x_0 = 0 # Initial value
t = 0  # Start time
a = 0   # Start time
b = 2  # End time





import numpy as np
import matplotlib.pyplot as plt

def x_n_func(x_n_2,x_n_1,h,function,t,n):



    x_n = x_n_2 + 2 * h * function(t + (n - 1)*h, x_n_1)
    return x_n

def bulirsch_stoer(x_0,function,t,h,N):
    x_list=[]
    x_list.append(x_0)
    x_1=x_0+function(t,x_0)*h
    x_list.append(x_1)
    for i in range(1,N-2):
        n=N-i
        x_n_2=x_list[i-1]
        x_n_1=x_list[i]
        x_n=x_n_func(x_n_2,x_n_1,h,function,t,n)
        x_list.append(x_n)
    H=h*N
    x_n=1/2 * (x_list[-1] + x_list[-2] + h * function(t+H, x_list[-1]))
    x_list.append(x_n)
    return x_list

# Define the derivative function
def f(t, x):
    return np.cos(t)



x_list = []
time_list = []

h_list = []
x_t_plus_h = []

N_list = []

N_start = 2

from scipy.interpolate import lagrange
lagrange_val=[]
lagrange_func=0
iterations=0
while True:
    iterations+=1
    print("Iteration: "+str(iterations))
    N_start = N_start*2
    h = np.float64((b - a) / N_start)
    x = bulirsch_stoer(x_0, f, t, h, N_start)


    t_list = np.linspace(a, b, N_start)

    #for plotting
    x_list.append(x)
    time_list.append(t_list)

    #last value of x
    x_t_plus_h.append(x[-1])

    N_list.append(N_start)
    h_list.append(h)
    if len(h_list) > 1:
        #calulate the lagrange interpolation
        x_lagrange = lagrange(h_list, x_t_plus_h)
        #print("The lagrange polynomial is: "+ str(x_lagrange))
        x_lagrange_0 = x_lagrange(0)
        if len(lagrange_val) > 1:
            if abs(x_lagrange_0 - lagrange_val[-1]) < 10**(-13):
                lagrange_final=x_lagrange
                break
            else:
                lagrange_val.append(x_lagrange_0)
        else:
            lagrange_val.append(x_lagrange_0)

print("The exact matimatical solution of the function sin({}) is: ".format(b)+str(np.sin(b)))
print("The numerical solution using the Langrange’s polynomial and Richardson extrapolation for max N={} with a difference of less than 10^-13 between two interpolations is: ".format(N_list[-1]) + str(lagrange_val[-1]))


for i in range(len(x_list)):
    plt.plot(time_list[i], x_list[i], label='N = {}'.format(N_list[i]))
#add the polinomial as a point
plt.plot(b, lagrange_val[-1], 'ro', label='Lagrange Interpolation')


time_list = np.linspace(a, b, 1000)
plt.plot(time_list, np.sin(time_list), label='True Function')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('sin({}): Bulirsch-Stoer Method for different N'.format(b))
plt.legend()
plt.show()
