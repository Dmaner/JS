# -*-coding:utf-8-*-
# method 1

from sympy.parsing.sympy_parser import parse_expr
from sympy import plot_implicit


ezplot = lambda exper: plot_implicit(parse_expr(exper))
f = 'y**3-3*x**2*y-1'
ezplot(f)

# method 2
import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)

x, y = np.meshgrid(x, y)
z = y**3-3*x**2*y-1
plt.contour(x, y, z, 0, cmap="Blues")
plt.show()