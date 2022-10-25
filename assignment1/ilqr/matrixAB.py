from sympy import *
import numpy as np

#values related to the physical structure
Lp, Mp, M, L, g = symbols("polemass_length masspole total_mass length gravity")

#control input
u = symbols("force")

#time instant
dt = symbols("dt")

#state variables
x, xdot, theta, thetadot = symbols("x x_dot theta theta_dot")

thetaddot = (g*sin(theta) - cos(theta)*((u + Lp*thetadot**2*sin(theta))/M)/L*(4/3 - (Mp*cos(theta)**2)/M))
xddot = (u + Lp * thetadot**2 * sin(thetadot) - Lp * thetaddot * cos(thetadot))/M

x_next = x + xdot*dt
x_dot_next = xdot + xddot*dt
theta_next = theta + thetadot*dt
theta_dot_next = thetadot + thetaddot*dt

a = Matrix([x, xdot, theta, thetadot])
b = Matrix([u])
f = Matrix([[x_next],[x_dot_next],[theta_next],[theta_dot_next]])
A = f.diff(a)
B = f.diff(b)

print(B)
print(B.shape)
print(A.shape)

C = np.array([[[1],[2]],[[3],[4]]])
print(C)
print(C.shape)

C = np.array([[1,2],[3,4]]).T
print(C)
print(C.shape)

# x1,x2,x3 = symbols("x1 x2 x3")
# u1,u2 = symbols("u1 u2")
# dt = symbols("dt")
# x = Matrix([[x1],[x2],[x3]])

# f = Matrix([[x1 + u1*cos(x3)*dt],
#            [x2 + u1*sin(x3)*dt],
#            [x3 + u2*dt]])

# print(np.array(f.diff(x)))
