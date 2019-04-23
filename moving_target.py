import numpy as np
import random
from environment import Terrain

DIM = 5
class Agent:
    def __init__(self, dim):
        self.dim = dim
        self.terrain = Terrain(dim)
        self.beliefs = np.full((dim, dim), (1/(dim*dim)))
        self.moves = 0

    def update_beliefs(self, cell_searched, cell_fn):
        cell_row = cell_searched[0]
        cell_col = cell_searched[1]

        # Fetch belief of cell at t-1
        prev_belief = self.beliefs[cell_row][cell_col]

        # Compute belief of cell at t
        new_belief = (cell_fn * prev_belief) / ((1-prev_belief) + prev_belief * cell_fn)

        # Compute difference in probabilities - will be proportionately distributed among rest of cells
        belief_difference = new_belief - prev_belief

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
    
    def get_neighbors(self, cell):
        #returns all neighbors of the current target
        neighbors = []
        tx, ty = cell
        
        if tx + 1 < self.dim-1:
            neighbors.append([tx + 1, ty])
        if tx - 1 > 0:
            neighbors.append([tx - 1, ty])
        if ty + 1 < self.dim-1:
            neighbors.append([tx, ty + 1])
        if ty - 1 > 0:
            neighbors.append([tx, ty - 1])
            
        return neighbors
            
    def move_target(self):
        #moves the target to one of the neighbors and keep track of the transition type
        
        self.terrain.transition.clear()
        
        tx, ty = self.terrain.target
        self.terrain.transition.add(self.terrain.board[tx][ty])
        
        self.terrain.target = random.choice(self.get_neighbors([tx, ty]))
        tx, ty = self.terrain.target
        self.terrain.transition.add(self.terrain.board[tx][ty])

    def update_transition_beliefs(self):
        #set probabilities of cells that dont match the transition as 0 and recalculate probability of cells that satisfy
        tra = self.terrain.transition
        total = 0.0
        temp_prob = np.zeros((self.dim, self.dim))
        for i in range(self.terrain.dim):
            for j in range(self.terrain.dim):
                
                cell_terrain = self.terrain.board[i][j]
                if cell_terrain in tra:
                    if len(tra) == 2:
                        tra.remove(cell_terrain)
                        
                    newprob = 0.0
                    for x, y in self.get_neighbors([i, j]):
                        c_ter = self.terrain.board[x][y]
                        if c_ter in tra:
                            newprob += self.beliefs[x][y]
                    temp_prob[i][j] = newprob
                    tra.add(cell_terrain)
                    
                else:
                    temp_prob[i][j] = 0
                
                total += temp_prob[i][j]
                
        self.beliefs = temp_prob / total
                

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
            self.move_target()
            self.update_transition_beliefs()
            self.solve()

if __name__== "__main__":
    Agent(DIM).solve()
