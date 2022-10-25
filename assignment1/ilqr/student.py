import numpy as np
from scipy import linalg
from numpy import sin, cos

class CartPole:
    def __init__(self, env, x=None,max_linear_velocity=2, max_angular_velocity=np.pi/3):
        if x is None:
            x = np.zeros(2)
        self.env = env
        
    def getA(self, u, x=None):
        if x is None:
            x = self.x
        x, x_dot, theta, theta_dot = x
        polemass_length = self.env.polemass_length
        gravity = self.env.gravity
        masspole = self.env.masspole
        total_mass = self.env.total_mass
        length = self.env.length
        force = u
        dt = self.env.tau
        
        A = np.eye(4)#dxd (sqare matrix)
        A = np.array([ [1,0,0,0], [dt,1,0,0], [0, -dt*polemass_length*(gravity*cos(theta) - 2*masspole*(force + polemass_length*theta_dot**2*sin(theta))*sin(theta)*cos(theta)**2/(length*total_mass**2) - polemass_length*theta_dot**2*(-masspole*cos(theta)**2/total_mass + 1.33333333333333)*cos(theta)**2/(length*total_mass) + (force + polemass_length*theta_dot**2*sin(theta))*(-masspole*cos(theta)**2/total_mass + 1.33333333333333)*sin(theta)/(length*total_mass))*cos(theta_dot)/total_mass, 1, dt*(gravity*cos(theta) - 2*masspole*(force + polemass_length*theta_dot**2*sin(theta))*sin(theta)*cos(theta)**2/(length*total_mass**2) - polemass_length*theta_dot**2*(-masspole*cos(theta)**2/total_mass + 1.33333333333333)*cos(theta)**2/(length*total_mass) + (force + polemass_length*theta_dot**2*sin(theta))*(-masspole*cos(theta)**2/total_mass + 1.33333333333333)*sin(theta)/(length*total_mass))], [0, dt*(polemass_length*theta_dot**2*cos(theta_dot) + 2*polemass_length*theta_dot*sin(theta_dot) + polemass_length*(gravity*sin(theta) - (force + polemass_length*theta_dot**2*sin(theta))*(-masspole*cos(theta)**2/total_mass + 1.33333333333333)*cos(theta)/(length*total_mass))*sin(theta_dot) + 2*polemass_length**2*theta_dot*(-masspole*cos(theta)**2/total_mass + 1.33333333333333)*sin(theta)*cos(theta)*cos(theta_dot)/(length*total_mass))/total_mass, dt, -2*dt*polemass_length*theta_dot*(-masspole*cos(theta)**2/total_mass + 1.33333333333333)*sin(theta)*cos(theta)/(length*total_mass) + 1]],dtype=float).T
        
        return A
        
    def getB(self, x=None):
        if x is None:
            x = self.x
        
        x, x_dot, theta, theta_dot = x
        polemass_length = self.env.polemass_length
        gravity = self.env.gravity
        masspole = self.env.masspole
        total_mass = self.env.total_mass
        length = self.env.length
        dt = self.env.tau

        B = np.zeros((4,1))#dxk (column vector)
        B = np.array([[0, dt*(1 + polemass_length*(-masspole*cos(theta)**2/total_mass + 1.33333333333333)*cos(theta)*cos(theta_dot)/(length*total_mass))/total_mass, 0, -dt*(-masspole*cos(theta)**2/total_mass + 1.33333333333333)*cos(theta)/(length*total_mass)]]).T

        return B

def lqr(A, B, T=100):
    #d = 4 and k = 1 as defined by the dynamical system (or transition function)

    K = np.zeros((1,4)) #gain matrix -> 1x4 (will be multiplied by the error (which is 4x1) giving a scalar)
    R = np.eye(1) #compliant to the control signal -> kxk -> 1x1
    Q = np.eye(4) #compliant to the state -> dxd -> 4x4
    P = np.zeros((4,4)) #compliant to the state -> dxd -> 4x4

    for t in range(T):

        K = -linalg.inv(R+B.T@P@B)@B.T@P@A
        P = (Q + K.T@R@K + (A+B@K).T@P@(A+B@K))

    return K