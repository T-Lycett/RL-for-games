import TDAgent
import math
import numpy as np
import checkersBoard
from scipy import stats


class MCTS:
    def __init__(self, nnet_model, use_policy_head=True):
        self.use_policy_head = use_policy_head
        self.cpuct = 0.1
        self.e = 0.75
        self.nnet_model = nnet_model
        self.Qs = {}
        self.v_lower_bound = {}
        self.v_upper_bound = {}
        self.Nsa = {}
        self.Ns = {}
        self.Es = {}
        self.Ps = {}
        self.mcts_sims = 0
        self.max_sims = None
        self.kld_threshold = None
        self.kl_divergence = None
        self.noise = None
        self.eval_batch_size = 5
        self.max_depth = 0

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

        self.v_lower_bound[(state)] = -math.inf
        self.v_upper_bound[(state)] = math.inf
        self.max_depth = 0
        while not self.terminate_search():
            self.mcts_sims += 1
            self.search(board, self.v_lower_bound[(state)], self.v_upper_bound[(state)], 0, dir_alpha)

            if self.mcts_sims % 25 == 0:
                #  print(self.mtcs_sims)
                new_probs = np.zeros(checkersBoard.CheckersBoard.action_size)
                counts = np.zeros(checkersBoard.CheckersBoard.action_size)
                q_values = np.zeros(checkersBoard.CheckersBoard.action_size)
                lower_bounds = []
                upper_bounds = []
                for move, index in valid_moves:
                    index = int(index)
                    move = TDAgent.extract_features(move, move.current_player).tobytes()
                    counts[index] = self.Nsa[(state, move)] if (state, move) in self.Nsa.keys() else 0
                    q_values[index] = self.Qs[(move)] if (move) in self.Qs.keys() else 0
                    lower_bounds.append(self.v_lower_bound[(move)])
                    upper_bounds.append(self.v_upper_bound[(move)])
                new_probs = [x/sum(counts) for x in counts]
                if node_probs is not None:
                    self.kl_divergence = stats.entropy(new_probs, node_probs)
                    # print(kl_divergence)
                node_probs = new_probs
                # print(counts)
                if verbose:
                    root_q = self.Qs[(state)]
                    root_lower = self.v_lower_bound[(state)]
                    root_upper = self.v_upper_bound[(state)]
                    print('root q: ' + str(root_q) + ' v-: ' + str(root_lower) + ' v+: ' + str(root_upper))
                    print('counts: ' + str([x for x in counts if x != 0]))
                    print('probabilities: ' + str([x for x in node_probs if x != 0]))
                    print('q values: ' + str([x for x in q_values if x != 0]))
                    print('lower bounds: ' + str(lower_bounds))
                    print('upper bounds: ' + str(upper_bounds))

        if verbose:
            print('prior probs: ' + str([x for x in self.Ps[state] if x != 0]))

        counts = np.zeros(checkersBoard.CheckersBoard.action_size)
        child_scores = np.zeros(checkersBoard.CheckersBoard.action_size)
        for move, index in valid_moves:
            move = TDAgent.extract_features(move, move.current_player).tobytes()
            counts[int(index)] = self.Nsa[(state, move)] if (state, move) in self.Nsa.keys() else 0
            child_scores[int(index)] = self.Qs[(move)] + 2 if (move) in self.Qs.keys() else -2

        if temperature == 0:
            best_move = np.argmax(child_scores)
            exponentiated_probs = [0]*len(counts)
            exponentiated_probs[best_move] = 1
        else:
            counts = [x**(1/temperature) for x in counts]
            exponentiated_probs = [x/float(sum(counts)) if x != 0 else 0 for x in counts]

        return exponentiated_probs, node_probs

    # def batch_search(self, root_node, dir_alpha=0):


    def search(self, board, alpha, beta, depth, dir_alpha=0):
        self.max_depth = max(self.max_depth, depth)
        depth = depth + 1

        current_player = board.current_player
        state = TDAgent.extract_features(board, current_player).tobytes()
        if state not in self.Ns:
            self.Ns[state] = 0

        ended, winner = board.game_ended()
        if ended:
            self.Qs[(state)] = self.v_upper_bound[(state)] = self.v_lower_bound[(state)] = winner
            return

        valid_moves = board.get_valid_moves(current_player, include_index=True, include_chain_jumps=False)
        if self.Ns[state] == 0:
            valids = np.zeros(checkersBoard.CheckersBoard.action_size)
            for move, idx in valid_moves:
                new_state = TDAgent.extract_features(move, move.current_player).tobytes()
                self.Ns[new_state] = 0
                self.v_lower_bound[(new_state)] = -math.inf
                self.v_upper_bound[(new_state)] = math.inf
                valids[int(idx)] = 1
            self.Ns[state] += 1
            ended, winner = board.game_ended()
            if ended:
                self.Qs[(state)] = self.v_lower_bound[(state)] = self.v_upper_bound[(state)] = winner
                return
            else:
                features = TDAgent.extract_features(board, current_player)
                features = np.asarray([features])
                if self.use_policy_head:
                    v, pi = self.nnet_model.predict(features)
                    v = v[0][0]
                    pi = pi[0]
                    if sum(valids) == 0:
                        print('error: no valid moves in list.')
                    else:
                        pi *= valids
                        pi = [x/float(sum(pi)) if x != 0 else 0 for x in pi]
                    self.Ps[state] = pi
                else:
                    v = self.nnet_model.predict(features)
                    v = v[0]
                    self.Ps[state] = np.ones(checkersBoard.CheckersBoard.action_size)
                self.Qs[(state)] = self.v_lower_bound[(state)] = self.v_upper_bound[(state)] = v * current_player
                return

        if current_player == 1:
            v_lower = -math.inf
            v_upper = -math.inf
            child_exists = False
            all_child_nodes_expanded = True
            for move, move_index in valid_moves:
                move_state = TDAgent.extract_features(move, move.current_player).tobytes()
                if (move_state) in self.Qs.keys():
                    v_lower = max(v_lower, self.v_lower_bound[(move_state)])
                    v_upper = max(v_upper, self.v_upper_bound[(state)])
                    child_exists = True
                else:
                    all_child_nodes_expanded = False
        else:
            v_lower = math.inf
            v_upper = math.inf
            child_exists = False
            all_child_nodes_expanded = True
            for move, move_index in valid_moves:
                move_state = TDAgent.extract_features(move, move.current_player).tobytes()
                if (move_state) in self.Qs.keys():
                    v_lower = min(v_lower, self.v_lower_bound[(move_state)])
                    v_upper = min(v_upper, self.v_upper_bound[(state)])
                    child_exists = True
                else:
                    all_child_nodes_expanded = False
        if child_exists and all_child_nodes_expanded:
            self.v_lower_bound[(state)] = v_lower
            self.v_upper_bound[(state)] = v_upper

        best_u = -math.inf
        best_move = None
        i = 0
        for move, move_index in valid_moves:
            move_index = int(move_index)
            move_state = TDAgent.extract_features(move, move.current_player).tobytes()
            alphac = max(alpha, self.v_lower_bound[(move_state)])
            betac = min(beta, self.v_upper_bound[(move_state)])
            if True: # alphac < betac:
                if (state, move_state) in self.Nsa.keys() and (move_state) in self.Qs.keys():
                    p = self.Ps[state][move_index]
                    if dir_alpha != 0 and self.use_policy_head:
                        p = (p * self.e) + self.noise[i] * (1 - self.e)
                    u = (self.Qs[(move_state)] * current_player) + self.cpuct * p * math.sqrt(self.Ns[state] / (1 + self.Nsa[(state, move_state)]))
                    # u = self.Qsa[(state, move_state)] + self.cpuct * math.sqrt(self.Ns[state] / (1 + self.Nsa[(state, move_state)]))
                else:
                    u = self.cpuct * math.sqrt(self.Ns[state]/1e-8)
                if u > best_u:
                    best_u = u
                    best_move = move
            i += 1

        if best_move is not None:
            move_state = TDAgent.extract_features(best_move, best_move.current_player).tobytes()
            self.search(best_move, self.v_lower_bound[(move_state)], self.v_upper_bound[(move_state)], depth)
        # if current_player != best_move.current_player:
            # v *= -1

        if (state, move_state) in self.Nsa.keys():
            # self.Qsa[(state, move_state)] = (self.Nsa[(state, move_state)] * self.Qsa[(state, move_state)] + v) / (self.Nsa[(state, move_state)] + 1)
            self.Nsa[(state, move_state)] += 1
        else:
            # self.Qsa[(state, move_state)] = v
            self.Nsa[(state, move_state)] = 1

        if current_player == 1:
            v_lower = -math.inf
            v_upper = -math.inf
            max_score = -1
            child_exists = False
            all_child_nodes_expanded = True
            for move, move_index in valid_moves:
                move_index = int(move_index)
                move_state = TDAgent.extract_features(move, move.current_player).tobytes()
                if (move_state) in self.Qs.keys():
                    max_score = max([max_score, self.Qs[move_state]])
                    child_exists = True
                else:
                    all_child_nodes_expanded = False
                v_lower = max(v_lower, self.v_lower_bound[(move_state)])
                v_upper = max(v_upper, self.v_upper_bound[(state)])
            if child_exists and all_child_nodes_expanded:
                self.Qs[(state)] = max_score
            elif child_exists and not all_child_nodes_expanded:
                if max_score > self.Qs[(state)]:
                    self.Qs[(state)] = max_score
        else:
            v_lower = math.inf
            v_upper = math.inf
            min_score = 1
            child_exists = False
            all_child_nodes_expanded = True
            for move, move_index in valid_moves:
                move_index = int(move_index)
                move_state = TDAgent.extract_features(move, move.current_player).tobytes()
                if (move_state) in self.Qs.keys():
                    min_score = min([min_score, self.Qs[move_state]])
                    child_exists = True
                else:
                    all_child_nodes_expanded = False
                v_lower = min(v_lower, self.v_lower_bound[(move_state)])
                v_upper = min(v_upper, self.v_upper_bound[(state)])
            if child_exists and all_child_nodes_expanded:
                self.Qs[(state)] = min_score
            elif child_exists and not all_child_nodes_expanded:
                if min_score < self.Qs[(state)]:
                    self.Qs[(state)] = min_score
        self.v_lower_bound[(state)] = v_lower
        self.v_upper_bound[(state)] = v_upper

        self.Ns[state] += 1
        return

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
