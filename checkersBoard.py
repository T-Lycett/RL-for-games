import numpy as np

class CheckersBoard():
    def __init__(self, start_positions = False, board = None):
        self.reset()
        if start_positions:
            self.setStartPositions()
        if board is not None:
            self.setPositions(board)

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

    def isValidPosition(self, row, column):
        if row >= 0 and row <= 7 and column >= 0 and column <= 7:
            return True
        return False

    def isEmptyPosition(self, row, column):
        if self.isValidPosition(row, column):
            if self.p1_positions[row, column] == 0 and self.p2_positions[row, column] == 0:
                return True
        return False

    def getPlayerPositions(self, player):
        if player == 1:
            return self.p1_positions
        if player == -1:
            return self.p2_positions
        return None

    def getPlayersKings(self, player):
        if player == 1:
            return self.p1_kings
        if player == -1:
            return self.p2_kings
        return None

    def isValidMove(self, player, start, end):
        if not self.isEmptyPosition(end[0], end[1]) or not self.isValidPosition(start[0], start[1]):
            return False
        player_positions = self.getPlayerPositions(player)
        player_kings = self.getPlayersKings(player)
        opponent_positions = self.getPlayerPositions(-(player))
        if player_positions[start[0], start[1]] == 0: #if no token at start
            return False
        if player_kings[start[0], start[1]] == 0:
            if start[0] - end[0] != player: #going in wrong direction
                return False
            if abs(start[1] - end[1]) != 1:
                return False
        else:
            if abs(start[0] - end[0]) != 1 or abs(start[1] - end[1] != 1): #is not diagonally adjacent
                return False
        return True

    def executeMove(self, player, start, end):
        if not self.isValidMove(player, start, end):
            return False
        players_positions = self.getPlayerPositions(player)
        players_kings = self.getPlayersKings(player)
        players_positions[start[0], start[1]] = 0
        players_positions[end[0], end[1]] = 1
        players_kings[end[0], end[1]] = players_kings[start[0], start[1]]
        players_kings[start[0], start[1]] = 0
        return True

    def getValidMovesFromPos(self, player, pos):
        valid_moves = []
        row_offset = -(player)
        player_positions = self.getPlayerPositions(player)
        players_kings = self.getPlayersKings(player)
        opponent_positions = self.getPlayerPositions(-(player))
        if player_positions[pos[0], pos[1]] == 1:
            x = pos[0] + row_offset
            y = pos[1] - 1
            if self.isEmptyPosition(x, y):
                new_move = CheckersBoard(board=self)
                if new_move.executeMove(player, pos, [x, y]):
                    valid_moves.append(new_move)
            y = pos[1] + 1
            if self.isEmptyPosition(x, y):
                new_move = CheckersBoard(board=self)
                if new_move.executeMove(player, pos, [x, y]):
                    valid_moves.append(new_move)
            if players_kings[pos[0], pos[1]] == 1:
                x = pos[0] - row_offset
                y = pos[1] - 1
                if self.isEmptyPosition(x, y):
                    new_move = CheckersBoard(board=self)
                    if new_move.executeMove(player, pos, [x, y]):
                        valid_moves.append(new_move)
                y = pos[1] + 1
                if self.isEmptyPosition(x, y):
                    new_move = CheckersBoard(board=self)
                    if new_move.executeMove(player, pos, [x, y]):
                        valid_moves.append(new_move)
        return valid_moves

    def getValidMoves(self, player):
        valid_moves = []
        player_positions = self.getPlayerPositions(player)
        players_kings = self.getPlayersKings(player)
        opponent_positions = self.getPlayerPositions(-(player))
        for row in range(player_positions.shape[0]):
            for column in range(player_positions.shape[1]):
                if player_positions[row, column] == 1:
                    new_moves =self.getValidMovesFromPos(player, [row, column])
                    if len(new_moves) > 0:
                        valid_moves = np.hstack((valid_moves, new_moves))
        return valid_moves
