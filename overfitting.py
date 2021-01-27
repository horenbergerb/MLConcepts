import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import random

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

'''
This is a demonstration of the capacity for overfitting by various loss functions.
In the quadratic regression, it appears there cannot be overfitting.
As the degree of the regression increases, overfitting becomes apparent.
'''

random.seed()

def noisy_quadratic(x):
    return (x**2)+random.uniform(-(x**2)/2,(x**2)/2)

def get_noisy_quadratic():
    x = np.array([cur_x for cur_x in np.arange(-10,10,.1)])
    y = np.array([noisy_quadratic(cur_x) for cur_x in x])
    return x,y

def polynomial_regression(x,y,degree=2):
    model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                      ('linear', LinearRegression(fit_intercept=False))])
    model = model.fit(x,y)
    predicted_y = model.predict(x)
    return predicted_y

x,y = get_noisy_quadratic()
plt.xlim(min(x)-1, max(x)+1)
plt.ylim(min(y)-1, max(y)+1)
plt.scatter(x,y)

x = x.reshape(-1,1)
y = y.reshape(-1,1)
print(x)

legend = [2,8,32,128]
for cur_degree in legend:
    predicted_y = polynomial_regression(x,y,degree=cur_degree)
    plt.plot(x,predicted_y)

plt.title("n-th degree regressions on a noisy quadratic")
plt.legend(legend)
plt.savefig("overfitting.png")
plt.show()
