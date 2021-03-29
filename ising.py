from typing import List
import numpy as np
from numpy.core.fromnumeric import mean, size
import random as rand
import math
import matplotlib.pyplot as plt


class Ising:
    # Constructor defines the board of size n and filled with 1's
    def __init__(self, size: int) -> None:
        self.size = size
        self.M = np.ones((size, size))

    # Gets the neighboring fields for a specific coordinate on the board.
    def get_neighbors(self, x: int, y: int) -> List[int]:
        neighbors = []  # list of neighbors
        mod = (size-1)
        for i in [-1, 1]:  # x-coordinate
            for j in [-1, 1]:  # y-coordinate
                # % mod will make the values go from the end to the beginning if there is a overflow
                neighbors.append(self.M[(x+i) % mod][(y+j) % mod])
        return neighbors

    def get_energy_for_coordinate(self, x: int, y: int):
        sum = 0  # sum of energy
        me = self.M[x][y]  # the coordinate's value
        # gets neighbors and loops throw them
        for neighbor in self.get_neighbors(x, y):
            # cool if statement to sum up the potential energy of the Ising model
            sum += 1 if me != neighbor else -1
        return sum

    # loops throw every coordinate and returns the summarized energies
    def get_energy(self) -> int:
        sum = 0
        for i in range(self.size-1):
            for j in range(self.size - 1):
                sum += self.get_energy_for_coordinate(i, j)
        return sum

    # toggels the value of a coordinate
    def toggle(self, x: int, y: int) -> None:
        self.M[x][y] = (-1)*self.M[x][y]

    # the monte-carlo step function
    def mc_steps(self, temp: float, iter: int) -> List[List[int]]:
        energies = []  # a list containing the energies of all iterations
        spins = []  # a list container the spins of all iterations

        E = self.get_energy()  # the total energy
        S = sum(sum(self.M))  # the total spin

        # run the monte-carlo simulations "iter" number of times
        for i in range(iter):
            x = rand.randint(0, self.size-1)  # random x-coordinate
            y = rand.randint(0, self.size-1)  # random y-coordinate

            Ei = self.get_energy_for_coordinate(
                x, y)  # gets the initial energy

            self.toggle(x, y)  # flips a spin

            Ef = self.get_energy_for_coordinate(x, y)  # gets the final energy

            dE = Ef-Ei  # calc the delta energy
            z = rand.random()  # pick a random float between 0 and 1

            # if delta energy is positive then check against the probability of the state
            # else accept the change
            if (0 < dE):
                if (z > math.exp(-dE/temp)):
                    # the change is rejected
                    self.toggle(x, y)
                    continue  # skips the last lines

            E += dE
            S += 2 * self.M[x][y]
            energies.append(E)
            spins.append(S)

        return [energies, spins]


size = 30  # size of board
I = Ising(size)  # initializing the model

# break the length 0 to 3 into "size" number of equal lengths. will be the z-axis
xs = np.linspace(0, 3, size)
ys = []  # initializing y's list

# temperature scan
for temperature in xs:
    # r contains the list of spins
    r = I.mc_steps(temperature, 200*size*size)[1]
    # dividing by size**2 (size to the power of two), to get the mean magnetization for every coordinate
    ys.append(mean(r)/size**2)  # adds the value to the y list
    print(math.floor(temperature/3*100), "%")

# plots
plt.plot(xs, ys)
plt.ylabel('<alpha>')
plt.xlabel('temp')
plt.show()
