import numpy as np
import matplotlib.pyplot as plt



def x_n(x_n_2,x_n_1,h,function,t,n):
    # x_n_2 is x_n-2
    # x_n_1 is x_n-1
    # h is the step size
    # function is the function to be integrated
    # t is the independent variable
    # n is the order of the method
    function=x_n_2+2*h*function(t+h)
    
    return 

def bulirsch_stoer(x_0,function,t,h):
    # function is the function to be integrated
    # t is the initial value of the independent variable
    
    x_0=x_0
    x_1=x_0+function(t+h)*h
    
