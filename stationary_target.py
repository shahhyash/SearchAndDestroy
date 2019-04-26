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

        print("[update_beliefs] Updating beliefs for failure observation on cell (%d, %d)" % (cell_searched[0], cell_searched[1]))
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

        while result is not True:
            # update beliefs based on existing check
            self.update_beliefs(to_check, to_check_fn)

            # update moves counter
            self.moves += 1

            # fetch coordinates of highest value in our belief matrix
            to_check = divmod(self.beliefs.argmax(), self.beliefs.shape[1])

            # fetch false neg probability of cell we are searching
            to_check_fn = self.terrain.get_cell_fn(to_check)

            # search that cell
            result = self.terrain.search_cell(to_check)

        print("[SOLVER]   Target found at cell (%d, %d). Took %d moves to find." % (to_check[0], to_check[1], self.moves))

    def fetch_most_probable_neighbor(self, cell):
        row = cell[0]
        col = cell[1]

        neighbors = [[row+1, col], [row-1, col], [row, col+1], [row, col-1]]
        neighbors_beliefs = np.zeros(4)

        for i in range(4):
            neighbor = neighbors[i]
            neighbor_row = neighbor[0]
            neighbor_col = neighbor[1]
            if neighbor_row >= 0 and neighbor_row < self.dim and neighbor_col >= 0 and neighbor_col < self.dim:
                neighbors_beliefs[i] = self.beliefs[neighbor_row][neighbor_col]

        index = neighbors_beliefs.argmax()
        return neighbors[index], neighbors_beliefs[index]

    def compute_manhattan_distance(self, cell1, cell2):
        return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])

    def solve_using_neighbors(self, prob_threshold=0.5):
        # This version of the solver agent adds a cost factor to moving a space on the grid.

        self.moves = 1

        # fetch coordinates of highest value in our belief matrix
        to_check = divmod(self.beliefs.argmax(), self.beliefs.shape[1])

        # fetch false neg probability of cell we are searching
        to_check_fn = self.terrain.get_cell_fn(to_check)

        # search that cell
        result = self.terrain.search_cell(to_check)

        while result is not True:
            # update beliefs based on existing check
            self.update_beliefs(to_check, to_check_fn)

            # fetch coordinates and false neg probability of highest value in our belief matrix
            max_cell = divmod(self.beliefs.argmax(), self.beliefs.shape[1])
            max_belief = self.beliefs[max_cell[0]][max_cell[1]]

            # fetch most probable neighbor
            neighbor_cell, neighbor_belief = self.fetch_most_probable_neighbor(to_check)

            # if the probability of finding the target at the max belief cell is larger than the neighbor 
            # AND if it's believed probability is greater than the threshold, 
            # then make the move to search that cell
            if max_belief > neighbor_belief and max_belief > prob_threshold:
                self.moves += self.compute_manhattan_distance(to_check, max_cell)
                to_check = max_cell
            else:
                self.moves += 1
                to_check = neighbor_cell

            # search that cell
            to_check_fn = self.terrain.get_cell_fn(to_check)
            result = self.terrain.search_cell(to_check)

        print("[SOLVER]   Target found at cell (%d, %d). Took %d moves to find." % (to_check[0], to_check[1], self.moves))

if __name__== "__main__":
    solver = Agent(DIM,RULE)
    solver.solve_using_neighbors()