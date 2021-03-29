import numpy as np
from numpy.core.fromnumeric import mean, size
import random as rand
import math
import matplotlib.pyplot as plt


class Ising:
    def __init__(self, size: int) -> None:
        self.size = size
        self.M = np.ones((size, size)) * (-1)

    def get_neighbors(self, x: int, y: int):
        n = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                n.append(self.M[i][j])
        return n

    def get_energy_point(self, x: int, y: int):
        sum = 0
        me = self.M[x][y]
        for n in self.get_neighbors(x, y):
            sum -= 1 if me != n else -1
        return sum

    def get_energy(self):
        sum = 0
        for i in range(self.size-1):
            for j in range(self.size - 1):
                sum += self.get_energy_point(i, j)
        return sum

    def toggle(self, x: int, y: int):
        self.M[x][y] = (-1)*self.M[x][y]

    def mc_steps(self, temp: float, inter: int, ):
        energies = []
        spins = []

        E = self.get_energy()
        S = sum(sum(self.M))

        for i in range(inter):
            x = rand.randint(0, self.size-1)
            y = rand.randint(0, self.size-1)

            Ei = self.get_energy_point(x, y)

            self.toggle(x, y)

            Ef = self.get_energy_point(x, y)

            dE = Ei - Ef
            z = rand.random()

            if (0 < dE):
                if (z < math.exp(-dE/temp)):
                    E += dE
                    S += 2 * self.M[x][y]
                    energies.append(E)
                    spins.append(S)
                else:
                    self.toggle(x, y)
            else:
                E += dE
                S += 2 * self.M[x][y]
                energies.append(E)
                spins.append(S)

        return [energies, spins]


size = 30
I = Ising(size)

xs = np.linspace(0.1, 3, size)
ys = []

# temp scan
for temp in xs:
    r = I.mc_steps(temp, 200*size*size)[1]
    ys.append(mean(r)/size**2)
    print(temp)

plt.plot(xs, ys)
plt.ylabel('<alpha>')
plt.xlabel('temp')
plt.show()
