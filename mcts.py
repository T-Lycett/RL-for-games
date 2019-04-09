import TDAgent
import math
import numpy as np
import checkersBoard


class MCTS:
    def __init__(self, nnet_model):
        self.cpuct = 0.3
        self.nnet_model = nnet_model
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Es = {}
        self.Ps = {}

    def get_probabilities(self, board, player, num_sims=25, temperature=1, verbose=False):

        for i in range(num_sims):
            self.search(board)
        state = TDAgent.extract_features(board, player).tobytes()
        valid_moves = board.get_valid_moves(player, include_index=True, include_chain_jumps=False)
        counts = np.zeros(checkersBoard.CheckersBoard.action_size)
        q_values = []
        for move, index in valid_moves:
            move = TDAgent.extract_features(move, move.current_player).tobytes()
            counts[int(index)] = self.Nsa[(state, move)] if (state, move) in self.Qsa.keys() else 0
            # q_values.append(self.Qsa[(state, move)])
        # print(counts)
        # print(q_values)

        if temperature == 0:
            best_move = np.argmax(counts)
            probs = [0]*len(counts)
            probs[best_move] = 1
            if sum(probs) > 1.01 or verbose:
                print('counts: ' + str(counts))
                print('probabilities: ' + str(probs))
            return probs
        else:
            counts = [x**(1/temperature) for x in counts]
            probs = [x/float(sum(counts)) if x != 0 else 0 for x in counts]
            if sum(probs) > 1.01 or verbose:
                print('counts: ' + str(counts))
                print('probabilities: ' + str(probs))
            return probs

    def search(self, board):

        current_player = board.current_player
        state = TDAgent.extract_features(board, current_player).tobytes()
        if state not in self.Ns:
            self.Ns[state] = 0

        ended, winner = board.game_ended()
        if ended:
            return winner * current_player

        valid_moves = board.get_valid_moves(current_player, include_index=True, include_chain_jumps=False)
        if self.Ns[state] == 0:
            # valids = np.zeros(checkersBoard.CheckersBoard.action_size)
            for move, idx in valid_moves:
                new_state = TDAgent.extract_features(move, move.current_player).tobytes()
                self.Ns[new_state] = 0
                # valids[int(idx)] = 1
            self.Ns[state] += 1
            ended, winner = board.game_ended()
            if ended:
                return winner * current_player
            else:
                features = TDAgent.extract_features(board, current_player)
                features = np.asarray([features])
                # v, pi = self.nnet_model.predict(features)
                v = self.nnet_model.predict(features)
                # pi = pi[0]
                # if sum(valids) == 0:
                    # print('error: no valid moves in list.')
                # else:
                    # pi *= valids
                    # pi = [x/float(sum(pi)) if x != 0 else 0 for x in pi]
                # self.Ps[state] = pi
                return v

        best_u = -math.inf
        best_move = None
        for move, move_index in valid_moves:
            move_index = int(move_index)
            move_state = TDAgent.extract_features(move, move.current_player).tobytes()
            if (state, move_state) in self.Qsa.keys():
                # u = self.Qsa[(state, move_state)] + self.cpuct * self.Ps[state][move_index] * math.sqrt(self.Ns[state] / (1 + self.Nsa[(state, move_state)]))
                u = self.Qsa[(state, move_state)] + self.cpuct * math.sqrt(self.Ns[state] / (1 + self.Nsa[(state, move_state)]))
            else:
                u = self.cpuct * math.sqrt(self.Ns[state]/1e-8)
            if u > best_u:
                best_u = u
                best_move = move

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
