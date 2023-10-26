import time

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt



def parabola(x_range = (-1,1),a=1,b=0,c=0,num_of_points = 100,vertix=None):
    '''
    Function of parabola, return Y mapped by equation.
    :param x: List of X coords.
    :param a: Coefficient that show down or up placed parabola branches
    :param b:
    :param c: Y
    :return:
    '''
    random.seed(seed=42)
    crossx = ((x_range[0],0),(x_range[1],0))#Roots of eq (x1,x2)
    tree_point = [crossx[0],vertix,crossx[1]]#3 points to find coefficients ((x1,y1),center (x0,y0), (x2,y2))
    x_1 = tree_point[0][0]
    x_2 = tree_point[1][0]
    x_3= tree_point[2][0]
    y_centr = tree_point[1][1]
    a =np.array([[x_1**2,x_1,1],[x_2**2,x_2,1],[x_3**2,x_3,1]])
    b =np.array([0,y_centr,0])
    coeff = np.linalg.solve(a,b)
    a,b,c = coeff
    X = np.linspace(x_range[0], x_range[1], num_of_points)
    y = [a*x**2+b*x+c for x in X]

    return [X,y]

par = parabola((-2,2),vertix=(0,-9))
#im_par = plt.plot(par[0],par[1])
#im1 =

plt.ion()

for _ in range(100):
    x = np.random.uniform(-4, 4)
    y = np.random.uniform(0, -5)
    par = parabola((-x, x), vertix=(0, y))
    plt.clf()
    plt.plot([-10, 10, 10, -10, -10], [0, 0, -10, -10, 0])
    plt.plot(par[0], par[1])
    plt.draw()
    plt.gcf().canvas.flush_events()

    #time.sleep(0.02)
plt.ioff()
#plt.show()


#plt.show()

class Eq_sampler:
    def __init__(self,equation,):
        pass
