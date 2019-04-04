import numpy as np
import random
from environment import Terrain

DIM = 20

class Agent:
    def __init__(self):
        self.terrain = Terrain(DIM)
        self.beliefs = np.full((DIM, DIM), (1/DIM))

    def solve(self):
        # fetch coordinates of highest value in our belief matrix
        to_check = divmod(self.beliefs.argmax(), self.beliefs.shape[1])
        self.terrain.search_cell(to_check)


if __name__== "__main__":
    Agent().solve()