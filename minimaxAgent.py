import math
import numpy as np

class MinimaxAgent:
    def __init__(self, player, max_depth=5):
        self.player = player #1 = player1, -1 = player 2
        self.max_depth = max_depth
        if self.player == 1:
            self.kings_row = 7
            self.opponent_kings_row = 0
        elif self.player == -1:
            self.kings_row = 0
            self.opponent_kings_row = 7

    def get_move(self, board):
        moves = board.get_valid_moves(self.player)
        num_moves = len(moves)
        if num_moves == 0:
            return None, None
        if num_moves == 1:
            return moves[0], self.evaluate(moves[0], 1)
        else:
            return self.minimax(board, 0)

    def minimax(self, board, current_depth=0, alpha=-math.inf, beta=math.inf):
        game_ended, _ = board.game_ended()
        if current_depth >= self.max_depth or game_ended:
            return board, self.evaluate(board, current_depth + 1)
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

    def evaluate(self, board, num_moves):
        has_ended, winner = board.game_ended()
        if has_ended:
            if winner == 0:
                return -2
            elif winner == self.player:
                return 100 - num_moves
            else:
                return -100 + num_moves

        players_pieces = board.get_players_pieces(self.player)
        opp_pieces = board.get_players_pieces(-self.player)

        score = 0
        max_dist = 0
        min_dist = 0
        for pos, king in players_pieces.items():
            score += 1 + (0.75 * king) + (abs(self.kings_row - pos[0]) * 0.1 * (1 - king))
            if king == 1:
                total_dist = 0
                for opp_pos, _ in board.get_players_pieces(-self.player).items():
                    total_dist += self.manhatten_dist(pos, opp_pos) * 0.05
                mean_dist = total_dist / len(opp_pieces)
                max_dist = max(max_dist, mean_dist)
                min_dist = min(min_dist, mean_dist)
        num_pieces = len(players_pieces)
        num_opp_peices = len(opp_pieces)
        if num_pieces > num_opp_peices:
            score -= max_dist
            opp_piece_val = 1.1
            score -= board.moves_without_capture * 0.01
        elif num_pieces == num_opp_peices:
            score -= max_dist
            opp_piece_val = 1.0
            score -= board.moves_without_capture * 0.01
        else:
            score += min_dist
            opp_piece_val = 0.9

        for pos, king in opp_pieces.items():
            score -= opp_piece_val + (0.75 * king) + (abs(self.opponent_kings_row - pos[0]) * 0.1 * (1 - king))


        return score

    def manhatten_dist(self, start, end):
        return abs(start[0] - end[0]) + abs(start[1] - end[1])
