import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

'''
This animation shows the gradient descent algorithm at work
'''

eta = .7
init_x = 8

def test_func(x):
    return x**2

# performs one step of gradient descent
def next_pt(prev_x, accur = .001):
    global eta
    grad = (test_func(prev_x+accur)-test_func(prev_x-accur))/(accur*2)
    next_x = prev_x-(grad*eta)
    return next_x, test_func(next_x)

fig, ax = plt.subplots()

# the function we are descending along
x = np.array([cur_x for cur_x in np.arange(-10,10,.1)])
y = np.array([test_func(cur_x) for cur_x in x])

# the points of our descent algorithm
x_pts = [init_x]
y_pts = [test_func(init_x)]

plt.xlim(min(x)-1, max(x)+1)
plt.ylim(min(y)-1, max(y)+1)

plt.plot(x,y)
pts = ax.scatter(x_pts, y_pts, color='r', s=15)
arrows = []

def update(frame):
    # only update every 15 frames
    # frame+1 so first frame doesn't update
    if (frame+1)%15 == 0:
        global x_pts
        global y_pts
        old_x = x_pts[len(x_pts)-1]
        old_y = y_pts[len(y_pts)-1]
        next_x, next_y = next_pt(old_x)
        # add an arrow between the newest point and the previous point
        arrow_scale = .04
        new_arrow = plt.arrow(old_x, old_y, next_x-old_x, next_y-old_y, color='k', width=arrow_scale, head_width=arrow_scale*3, head_length=arrow_scale*3*1.5)
        ax.add_patch(new_arrow)
        arrows.append(new_arrow)
        # update our list of points
        x_pts.append(next_x)
        y_pts.append(next_y)
        pts.set_offsets(zip(x_pts, y_pts))

        return tuple(arrows + [pts])
    return tuple(arrows + [pts])

ani = FuncAnimation(fig, update, frames=len(x)-2, interval=50,
                    blit=True)

# i use imagemagick because the default writer throws a weird error...
ani.save("grad_descent.gif", writer='imagemagick')

plt.show()
