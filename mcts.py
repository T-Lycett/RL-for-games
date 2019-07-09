import TDAgent
import math
import numpy as np
import checkersBoard
from scipy import stats


class MCTS:
    def __init__(self, nnet_model):
        self.cpuct = 0.5
        self.e = 0.75
        self.nnet_model = nnet_model
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Es = {}
        self.Ps = {}
        self.mcts_sims = 0
        self.max_sims = None
        self.kld_threshold = None
        self.kl_divergence = None
        self.noise = None

    def get_probabilities(self, board, player, kld_threshold, max_sims=None, temperature=1, verbose=False, dir_alpha=0):
        self.mcts_sims = 0
        self.max_sims = max_sims
        node_probs = None
        self.kl_divergence = math.inf
        self.kld_threshold = kld_threshold
        state = TDAgent.extract_features(board, player).tobytes()
        valid_moves = board.get_valid_moves(player, include_index=True, include_chain_jumps=False)
        if dir_alpha > 0:
            self.noise = np.random.dirichlet([dir_alpha] * len(valid_moves))
        while not self.terminate_search():
            self.mcts_sims += 1
            self.search(board, dir_alpha)

            if self.mcts_sims % 20 == 0:
                #  print(self.mtcs_sims)
                new_probs = np.zeros(checkersBoard.CheckersBoard.action_size)
                counts = np.zeros(checkersBoard.CheckersBoard.action_size)
                q_values = np.zeros(checkersBoard.CheckersBoard.action_size)
                for move, index in valid_moves:
                    index = int(index)
                    move = TDAgent.extract_features(move, move.current_player).tobytes()
                    counts[index] = self.Nsa[(state, move)] if (state, move) in self.Qsa.keys() else 0
                    q_values[index] = self.Qsa[(state, move)] if (state, move) in self.Qsa.keys() else 0
                new_probs = [x/sum(counts) for x in counts]
                if node_probs is not None:
                    self.kl_divergence = stats.entropy(new_probs, node_probs)
                    # print(kl_divergence)
                node_probs = new_probs
                # print(counts)
                if verbose:
                    print('counts: ' + str([x for x in counts if x != 0]))
                    print('probabilities: ' + str([x for x in node_probs if x != 0]))
                    print('q values: ' + str([x for x in q_values if x != 0]))

        if verbose:
            print('prior probs: ' + str([x for x in self.Ps[state] if x != 0]))

        counts = np.zeros(checkersBoard.CheckersBoard.action_size)
        for move, index in valid_moves:
            move = TDAgent.extract_features(move, move.current_player).tobytes()
            counts[int(index)] = self.Nsa[(state, move)] if (state, move) in self.Qsa.keys() else 0

        if temperature == 0:
            best_move = np.argmax(counts)
            exponentiated_probs = [0]*len(counts)
            exponentiated_probs[best_move] = 1
        else:
            counts = [x**(1/temperature) for x in counts]
            exponentiated_probs = [x/float(sum(counts)) if x != 0 else 0 for x in counts]

        return exponentiated_probs, node_probs

    def search(self, board, dir_alpha=0):

        current_player = board.current_player
        state = TDAgent.extract_features(board, current_player).tobytes()
        if state not in self.Ns:
            self.Ns[state] = 0

        ended, winner = board.game_ended()
        if ended:
            return winner * current_player

        valid_moves = board.get_valid_moves(current_player, include_index=True, include_chain_jumps=False)
        if self.Ns[state] == 0:
            valids = np.zeros(checkersBoard.CheckersBoard.action_size)
            for move, idx in valid_moves:
                new_state = TDAgent.extract_features(move, move.current_player).tobytes()
                self.Ns[new_state] = 0
                valids[int(idx)] = 1
            self.Ns[state] += 1
            ended, winner = board.game_ended()
            if ended:
                return winner * current_player
            else:
                features = TDAgent.extract_features(board, current_player)
                features = np.asarray([features])
                v, pi = self.nnet_model.predict(features)
                # v = self.nnet_model.predict(features)
                v = v[0][0]
                pi = pi[0]
                if sum(valids) == 0:
                    print('error: no valid moves in list.')
                else:
                    pi *= valids
                    pi = [x/float(sum(pi)) if x != 0 else 0 for x in pi]
                self.Ps[state] = pi
                return v

        best_u = -math.inf
        best_move = None
        i = 0
        for move, move_index in valid_moves:
            move_index = int(move_index)
            move_state = TDAgent.extract_features(move, move.current_player).tobytes()
            if (state, move_state) in self.Qsa.keys():
                p = self.Ps[state][move_index]
                if dir_alpha != 0:
                    p = (p * self.e) + self.noise[i] * (1 - self.e)
                u = self.Qsa[(state, move_state)] + self.cpuct * p * math.sqrt(self.Ns[state] / (1 + self.Nsa[(state, move_state)]))
                # u = self.Qsa[(state, move_state)] + self.cpuct * math.sqrt(self.Ns[state] / (1 + self.Nsa[(state, move_state)]))
            else:
                u = self.cpuct * math.sqrt(self.Ns[state]/1e-8)
            if u > best_u:
                best_u = u
                best_move = move
            i += 1

        v = self.search(best_move)
        if current_player != best_move.current_player:
            v *= -1

        move_state = TDAgent.extract_features(best_move, best_move.current_player).tobytes()
        if (state, move_state) in self.Qsa.keys():
            self.Qsa[(state, move_state)] = (self.Nsa[(state, move_state)] * self.Qsa[(state, move_state)] + v) / (self.Nsa[(state, move_state)] + 1)
            self.Nsa[(state, move_state)] += 1
        else:
            self.Qsa[(state, move_state)] = v
            self.Nsa[(state, move_state)] = 1

        self.Ns[state] += 1
        return v

    def terminate_search(self):
        if self.max_sims is not None:
            if self.max_sims <= self.mcts_sims:
                return True
            else:
                return False
        else:
            if self.kld_threshold >= self.kl_divergence:
                return True
            else:
                return False
