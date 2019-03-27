import numpy as np
import random
from environment import Terrain

DIM = 20

class Agent:
    def __init__(self):
        self.terrain = Terrain(DIM)
        self.beliefs = np.full((DIM, DIM), (1/DIM))

if __name__== "__main__":
    Agent()