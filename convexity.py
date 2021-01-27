import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

'''
This animation shows the 1st degree Taylor series expansion at every point of a function.
If a function is convex, this line should always be completely below the function.
Thus we can perform a visual inspection of whether functions are convex.
'''


def test_func(x):
    return x**2

def tangent_line(x, y, index):
    m = (y[index+1]-y[index-1])/(x[index+1]-x[index-1])
    result = [m*(cur_x-x[index])+y[index] for cur_x in x]
    return result

fig, ax = plt.subplots()

x = np.array([cur_x for cur_x in np.arange(-10,10,.1)])
y = np.array([test_func(cur_x) for cur_x in x])

plt.xlim(min(x)-1, max(x)+1)
plt.ylim(min(y)-1, max(y)+1)

plt.plot(x,y)
ln, = plt.plot(x, tangent_line(x,y,1))

def update(frame):
    global x
    global y
    ln.set_data(x, tangent_line(x,y,frame+1))
    return ln,

ani = FuncAnimation(fig, update, frames=len(x)-2, interval=50,
                    blit=True)

ani.save("quadratic_convexity.gif")

plt.show()
