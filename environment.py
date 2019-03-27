import matplotlib.pylab as plt
import numpy as np

CELL_FLAT=3         # represented by the color WHITE
CELL_HILLY=2        # represented by the color LIGHT GREEN
CELL_FORESTED=1     # represented by the color DARK GREEN
CELL_CAVES=0        # represented by the color BLACK

# Probabilities of existence; used for terrain generation
P_FLAT=0.2
P_HILLY=0.3
P_FORESTED=0.3
P_CAVES=0.2

class Terrain:
    def __init__(self, dim):
        self.dim = dim
        self.generate_board()
        self.place_target()
        self.visualize()

    def generate_board(self):
        self.board = np.random.choice([CELL_FLAT, CELL_HILLY, CELL_FORESTED, CELL_CAVES], (self.dim,self.dim), p=[P_FLAT, P_HILLY, P_FORESTED, P_CAVES])

    def place_target(self):
        self.target = np.random.choice(self.dim, 2)

    def visualize(self):
        plt.style.use('ggplot')
        plt.rcParams["axes.axisbelow"] = False

        fig = plt.figure(figsize = (5,5))
        ax = fig.add_subplot(111)

        ax.pcolormesh(self.board, cmap='gist_earth', zorder=1) 
        ax.grid(True, color="black", lw=1)           

        # set range of ticks to show entire grid
        ticks = np.arange(0, self.board.shape[0], 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        # remove ticks
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.set_aspect('equal') #set the x and y axes to the same scale
        ax.invert_yaxis() #invert the y-axis so the first row of data is at the top3

        plt.show()