import TDAgent
import math
import numpy as np


class MCTS:
    def __init__(self, nnet_model):
        self.cpuct = 1
        self.nnet_model = nnet_model
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Es = {}

    def get_probabilities(self, board, player, num_sims=25, temperature=1):

        for i in range(num_sims):
            self.search(board, player)
        state = TDAgent.extract_features(board, player).tobytes()
        valid_moves = board.get_valid_moves(player)
        counts = []
        for move in valid_moves:
            move = TDAgent.extract_features(move, -player).tobytes()
            counts.append(self.Nsa[(state, move)])

        if temperature == 0:
            best_move = np.argmax(counts)
            probs = [0]*len(counts)
            probs[best_move] = 1
            return probs
        else:
            counts = [x**(1/temperature) for x in counts]
            probs = [x/float(sum(counts)) for x in counts]
            return probs

    def search(self, board, player):

        state = TDAgent.extract_features(board, player).tobytes()
        if state not in self.Ns:
            self.Ns[state] = 0

        ended, winner = board.game_ended()
        if ended:
            return -(winner * player)

        if self.Ns[state] == 0:
            valid_moves = board.get_valid_moves(player)
            for move in valid_moves:
                new_state = TDAgent.extract_features(move, -player).tobytes()
                self.Ns[new_state] = 0
            self.Ns[state] += 1
            ended, winner = board.game_ended()
            if ended:
                return -(winner * player)
            else:
                features = TDAgent.extract_features(board, player)
                features = np.asarray([features])
                return -self.nnet_model.predict(features)

        valid_moves = board.get_valid_moves(player)
        best_u = -math.inf
        best_move = None

        for move in valid_moves:
            move_state = TDAgent.extract_features(move, -player).tobytes()
            if (state, move_state) in self.Qsa.keys():
                u = self.Qsa[(state, move_state)] + self.cpuct * math.sqrt(self.Ns[state] / (1 + self.Nsa[(state, move_state)]))
            else:
                u = self.cpuct * math.sqrt(self.Ns[state]/1e-8)
            if u > best_u:
                best_u = u
                best_move = move

        v = self.search(best_move, -player)

        move_state = TDAgent.extract_features(best_move, -player).tobytes()
        if (state, move_state) in self.Qsa.keys():
            self.Qsa[(state, move_state)] = (self.Nsa[(state, move_state)] * self.Qsa[(state, move_state)] + v) / (self.Nsa[(state, move_state)] + 1)
            self.Nsa[(state, move_state)] += 1
        else:
            self.Qsa[(state, move_state)] = v
            self.Nsa[(state, move_state)] = 1

        self.Ns[state] += 1
        return -v
