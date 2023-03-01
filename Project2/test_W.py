r=0.0075
h=0.002*5
a_d=1/h
print("a_d",a_d)
R=r/h
print("R",R)
dx=r
print("dx",dx)
print(a_d * (-2 + 3/2 * R) * dx / h**2)