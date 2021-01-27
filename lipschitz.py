import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

'''
This animation shows the "Lipschitz cone" at every point of a function.
If a function is G-Lipschitz, it should never enter the cone.
Thus we can perform a visual inspection of whether functions are Lipschitz continuous.
'''

G = 1

def test_func(x):
    return np.sqrt(x**2 + 5)

def cone_line1(x, y, index):
    global G
    result = [G*(cur_x-x[index])+y[index] for cur_x in x]
    return result

def cone_line2(x, y, index):
    global G
    result = [-G*(cur_x-x[index])+y[index] for cur_x in x]
    return result


fig, ax = plt.subplots()

x = np.array([cur_x for cur_x in np.arange(-10,10,.1)])
y = np.array([test_func(cur_x) for cur_x in x])

plt.xlim(min(x)-1, max(x)+1)
plt.ylim(min(y)-1, max(y)+1)

plt.plot(x,y)
cone_ln1, = plt.plot(x, cone_line1(x,y,1), 'g')
cone_ln2, = plt.plot(x, cone_line2(x,y,1), 'g')

def update(frame):
    global x
    global y
    cone_ln1.set_data(x, cone_line1(x,y,frame+1))
    cone_ln2.set_data(x, cone_line2(x,y,frame+1))

    return cone_ln1, cone_ln2,

ani = FuncAnimation(fig, update, frames=len(x)-2, interval=50,
                    blit=True)

ani.save("lipschitz_test.gif")

plt.show()
