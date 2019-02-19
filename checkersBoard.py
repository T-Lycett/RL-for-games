import numpy as np

class CheckersBoard():
    def __init__(self, start_positions = False, board = None):
        self.reset()
        if start_positions:
            self.setStartPositions()

    def reset(self):
        self.p1_positions = np.zeros((8, 8))
        self.p2_positions = np.zeros((8, 8))
        self.p1_kings = np.zeros((8, 8))
        self.p2_kings = np.zeros((8, 8))

    def setStartPositions(self):
        self.reset()
        self.p1_positions[7, ::2] = 1
        self.p1_positions[6, :] = 1
        self.p1_positions[6, ::2] = 0
        self.p2_positions[0, :] = 1
        self.p2_positions[0, ::2] = 0
        self.p2_positions[1, ::2] = 1

    def setPositions(self, board):
        self.p1_positions = board.p1_positions.copy()
        self.p2_positions = board.p2_positions.copy()
        self.p1_kings = board.p1_kings.copy()
        self.p2_kings = board.p2_kings.copy()

