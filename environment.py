import matplotlib.pyplot as plt
import numpy as np
import random

CELL_FLAT=3         # represented by the color WHITE
CELL_HILLY=2        # represented by the color LIGHT GREEN
CELL_FORESTED=1     # represented by the color DARK GREEN
CELL_CAVES=0        # represented by the color BLACK

# Probabilities of existence; used for terrain generation
P_FLAT=0.2
P_HILLY=0.3
P_FORESTED=0.3
P_CAVES=0.2

# False Negative Search Probabilities
P_FLAT_falseneg = 0.1
P_HILLY_falseneg = 0.3
P_FORESTED_falseneg = 0.7
P_CAVES_falseneg = 0.9

def terrain_str(type):
    if type == CELL_FLAT:
        return 'FLAT'
    elif type == CELL_HILLY:
        return 'HILLY'
    elif type == CELL_FORESTED:
        return 'FORESTED'
    else:
        return 'CAVES'

class Terrain:
    def __init__(self, dim):
        self.dim = dim
        self.generate_board()
        self.place_target()
        print("[TERRAIN]  Board initialized with target located at (%d, %d)." % (self.target[0], self.target[1]))
        self.visualize()

    def generate_board(self):
        self.board = np.random.choice([CELL_FLAT, CELL_HILLY, CELL_FORESTED, CELL_CAVES], (self.dim,self.dim), p=[P_FLAT, P_HILLY, P_FORESTED, P_CAVES])

    # Uniformly Random coordinate selection for target 
    def place_target(self):
        self.target = np.random.choice(self.dim, 2)

    # Search cell for target given coordinates. Returns false if cell wasn't found/false negative, true if found.
    def search_cell(self, cell):
        print("\n[TERRAIN]  Searching cell with coordinates (%d, %d). Current target is (%d, %d)." % (cell[0], cell[1], self.target[0], self.target[1]))

        # Fetch cell terrain
        cell_terrain = self.board[cell[0]][cell[1]]
        print("[TERRAIN]  Searching cell with terrain type: %s." % terrain_str(cell_terrain))

        # Check if cell contains the target; if not, just return False
        if cell[0] == self.target[0] and cell[1] == self.target[1]:
            # Fetch false neg probabilities for terrain
            if cell_terrain == CELL_FLAT:
                p_fneg = P_FLAT_falseneg
            elif cell_terrain == CELL_HILLY:
                p_fneg = P_HILLY_falseneg
            elif cell_terrain == CELL_FORESTED:
                p_fneg = P_FORESTED_falseneg
            else:
                p_fneg = P_CAVES_falseneg

            # If random value is less than fneg value, then return False, otherwise return True
            if random.random() < p_fneg:
                return False
            else:
                return True
        else:
            return False
        

    def visualize(self):
        plt.style.use('ggplot')
        plt.rcParams["axes.axisbelow"] = False

        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111)

        ax.pcolormesh(self.board, cmap='gist_earth', zorder=1) 
        ax.grid(True, color="black", lw=1)           

        # set range of ticks to show entire grid
        ticks = np.arange(0, self.board.shape[0], 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        # remove ticks
        # ax.xaxis.set_ticklabels([])
        # # ax.yaxis.set_ticklabels([])
        # ax.xaxis.set_ticks_position('none')
        # ax.yaxis.set_ticks_position('none')

        ax.set_aspect('equal') #set the x and y axes to the same scale
        ax.invert_yaxis() #invert the y-axis so the first row of data is at the top3

        plt.text(0.5, -0.1, 'Target Location at (%d, %d)' % (self.target[0], self.target[1]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        plt.show()