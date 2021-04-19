from typing import Callable
import random as rand
import math


def SA(func: Callable[[float], float], temp0: float, step_size0: float, x0: float, iter: int, ):

    x = x0
    x_min = x
    y_min = func(x)
    temp = temp0
    step_size = step_size0

    for i in range(iter):
        dX = rand.choice([-1, 1])*step_size

        yi = func(x)
        yf = func(x + dX)
        dY = yf - yi
        if (0 < dY):
            if (rand.random() > math.exp(-dY/(temp))):
                # change is rejected
                continue
        if (yf < y_min):
            y_min = yf
            x_min = x

        x += dX
        temp = temp0*(1-i/iter)
        step_size = step_size0*rand.random()

    return [x_min, y_min]


def func(x): return -x*math.sin(math.pi*x)**2*math.exp(-3/20*x**2)+1/10*x**2


print(SA(func, temp0=1, step_size0=0.1, x0=3, iter=1000))
