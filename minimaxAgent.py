import math
import numpy as np

class MinimaxAgent:
    def __init__(self, player, max_depth=5):
        self.player = player #1 = player1, -1 = player 2
        self.max_depth = max_depth
        if self.player == 1:
            self.home_row = 7
            self.promotion_row = 0
        elif self.player == -1:
            self.home_row = 0
            self.promotion_row = 7

    def get_move(self, board):
        moves = board.get_valid_moves(self.player)
        num_moves = len(moves)
        if num_moves == 0:
            return None, None
        if num_moves == 1:
            return moves[0], self.evaluate(moves[0])
        else:
            return self.minimax(board, 0)

    def minimax(self, board, current_depth=0, alpha=-math.inf, beta=math.inf):
        game_ended, _ = board.game_ended()
        if current_depth >= self.max_depth or game_ended:
            return board, self.evaluate(board)
        if current_depth % 2 == 0:
            current_player = self.player
        else:
            current_player = -self.player
        actions = board.get_valid_moves(current_player)
        if current_player == self.player:
            best_score = -math.inf
            best_action_index = -1
            for i, a in enumerate(actions):
                _, score = self.minimax(a, current_depth + 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action_index = i
                    alpha = best_score
                if alpha >= beta:
                    break
            return actions[best_action_index], best_score
        else:
            best_score = math.inf
            best_action_index = -1
            for i, a in enumerate(actions):
                _, score = self.minimax(a, current_depth + 1, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_action_index = i
                    beta = best_score
                if alpha >= beta:
                    break
            return actions[best_action_index], best_score

    def evaluate(self, board):
        has_ended, winner = board.game_ended()
        if has_ended:
            return (100 * self.player * winner) - 2

        players_pieces = board.get_players_pieces(self.player)
        opp_pieces = board.get_players_pieces(-self.player)

        score = 0
        dist_score = 0
        for pos, king in players_pieces.items():
            score += 1 + (0.75 * king) + (abs(self.home_row - pos[0]) * 0.1 * (1 - king))
            if king == 1:
                for opp_pos, _ in board.get_players_pieces(-self.player).items():
                    dist_score += self.manhatten_dist(pos, opp_pos) * 0.01
        if len(players_pieces) >= len(opp_pieces):
            score -= dist_score / ((len(players_pieces)) * len(opp_pieces))
            opp_piece_val = 1.1
            score -= board.moves_without_capture * 0.01
        else:
            score += dist_score / ((len(players_pieces)) * len(opp_pieces))
            opp_piece_val = 1.0

        for pos, king in opp_pieces.items():
            score -= opp_piece_val + (0.75 * king) + (abs(self.promotion_row - pos[0]) * 0.1 * (1 - king))


        return score

    def manhatten_dist(self, start, end):
        return abs(start[0] - end[0]) + abs(start[1] - end[1])
