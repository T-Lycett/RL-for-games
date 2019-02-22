import checkersBoard
import numpy as np

class MinimaxAgent:
    def __init__(self, player, max_depth=5):
        self.player = player #1 = player1, -1 = player 2
        self.max_depth = max_depth

    def get_move(self, board):
        moves = board.get_valid_moves(self.player)
        num_moves = len(moves)
        if num_moves == 0:
            return None, None
        if num_moves == 1:
            return moves[0], self.evaluate(moves[0])
        else:
            return self.minimax(board, 0)

    def minimax(self, board, current_depth=0):
        game_ended, _ = board.game_ended()
        if current_depth >= self.max_depth or game_ended:
            return board, self.evaluate(board)
        if current_depth % 2 == 0:
            current_player = self.player
        else:
            current_player = -self.player
        actions = board.get_valid_moves(current_player)
        scores = np.ndarray([len(actions)])
        for i, a in enumerate(actions):
            _, scores[i] = self.minimax(a, current_depth + 1)
        if current_player == self.player:
            return actions[np.argmax(scores)], np.max(scores)
        else:
            return actions[np.argmin(scores)], np.min(scores)


    def evaluate(self, board):
        has_ended, winner = board.game_ended()
        if has_ended:
            return 100 * self.player * winner

        score = 0
        for n in board.get_player_positions(self.player).flat:
            score += n
        for n in board.get_player_positions(-self.player).flat:
            score -= n
        for k in board.get_players_kings(self.player).flat:
            score += k * 0.5
        for k in board.get_players_kings(-self.player).flat:
            score -= k * 0.5

        return score
