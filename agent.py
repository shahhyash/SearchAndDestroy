import numpy as np
import random
from environment import Terrain

# Dimension of Terrain
DIM = 10

# RULE=1 when search occurs at cell with the highest probability of _containing_ the target
# RULE=2 when search occurs at cell with the highest probability of _finding_ the target
RULE = 1

class Agent:
    def __init__(self, dim, rule):
        self.dim = dim
        self.terrain = Terrain(dim)

        self.beliefs = np.full((dim, dim), (1/(dim*dim)))
        if rule is not 1:
            for row in range(dim):
                for col in range(dim):
                    terrain_prob = 1 - self.terrain.get_cell_fn([row,col])
                    self.beliefs[row][col] = self.beliefs[row][col] * terrain_prob

        self.moves = 0

    def update_beliefs(self, cell_searched, cell_fn):
        cell_row = cell_searched[0]
        cell_col = cell_searched[1]

        # Fetch belief of cell at t-1
        prev_belief = self.beliefs[cell_row][cell_col]

        # Compute belief of cell at t
        new_belief = (cell_fn * prev_belief) / ((1-prev_belief) + prev_belief * cell_fn)

        # Compute difference in probabilities - will be proportionately distributed among rest of cells
        belief_difference = prev_belief - new_belief

        # Store new belief at cell location
        self.beliefs[cell_row][cell_col] = new_belief

        print("[update_beliefs] Updating beliefs for failure observation on cell (%d, %d)" % cell_searched)
        print("[update_beliefs] cell_fn=%.3g, prev_belief=%.3g, new_belief=%.3g, diff=%.3g" % (cell_fn, prev_belief, new_belief, belief_difference))

        # divide difference among rest of cells proportionately
        for row in range(self.dim):
            for col in range(self.dim):
                if row is not cell_row and col is not cell_col:
                    proportion = self.beliefs[row][col] / (1 - prev_belief)
                    self.beliefs[row][col] = self.beliefs[row][col] + (proportion * belief_difference)


    def solve(self):
        self.moves += 1

        # fetch coordinates of highest value in our belief matrix
        to_check = divmod(self.beliefs.argmax(), self.beliefs.shape[1])

        # fetch false neg probability of cell we are searching
        to_check_fn = self.terrain.get_cell_fn(to_check)

        # search that cell
        result = self.terrain.search_cell(to_check)

        if result:
            print("[SOLVER]   Target found at cell (%d, %d). Took %d moves to find." % (to_check[0], to_check[1], self.moves))
        else:
            self.update_beliefs(to_check, to_check_fn)
            self.solve()

if __name__== "__main__":
    Agent(DIM,RULE).solve()