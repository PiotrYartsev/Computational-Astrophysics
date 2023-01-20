import tqdm as tqdm


#get gravitational constant
G=6.67408*10**(-11)


Sun=[1,0,0,0,0,0,0]
Jupiter=[0.001,0,5.2,0,-2.75674,0,0]
Trojan1=[0,-4.503,2.6,0,-1.38,-2.39,0]
Trojan2=[0,-4.503,2.6,0,-1.38,2.39,0]

W=[Sun,Jupiter,Trojan1,Trojan2]


#improt lamda






#Get the force from one object on another
def function_one_iteration(G,W,i,j):
    object_i=W[i]
    object_j=W[j]
    x_i=object_i[1]
    y_i=object_i[2]
    x_j=object_j[1]
    y_j=object_j[2]
    r=((x_i-x_j)**2+(y_i-y_j)**2)**(1/2)
    g=G*object_j[0]/r**2
    g_x=g*(x_i-x_j)/r
    g_y=g*(y_i-y_j)/r
    return([g_x,g_y])

def Force_function(G,W):
    Force=[0,0]
    for i in range(len(W)):
        for j in range(len(W)):
            if i!=j:
                g_x,g_y=function_one_iteration(G,W,i,j)
                Force[0]=Force[0]+g_x
                Force[1]=Force[1]+g_y
    return(Force)


def fourth_order_runge_kutta(function, current_time, timestep, W):
    f_a=function(current_time, W)
    W_b=W+timestep*f_a/2
    f_b=function(current_time+timestep/2, W_b)
    W_c=W+timestep*f_b/2
    f_c=function(current_time+timestep/2, W_c)
    W_d=W+timestep*f_c
    f_d=function(current_time+timestep, W_d)
    return W+timestep*(f_a+2*f_b+2*f_c+f_d)/6


timestep=0.01
current_time=0

total_time=300


change_in_position=[]
change_in_position.append(W)
while current_time<total_time:
    W=fourth_order_runge_kutta(Force_function, current_time, timestep, W)
    current_time=current_time+timestep
    change_in_position.append(W)

for i in change_in_position:
    print(i[0])







    