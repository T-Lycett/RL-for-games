import TDAgent
import math
import numpy as np
import torch
import checkersBoard
from scipy import stats


class MCTS:
    """
    Implements a Monte Carlo Tree Search algorithm enhanced with neural network guidance
    and potentially features inspired by Transposition-Enhanced MCTS (TEMCTS).
    It uses KL-divergence for dynamic search termination.
    """
    def __init__(self, nnet_model, use_policy_head=True):
        """
        Initializes the MCTS tree and parameters.

        Args:
            nnet_model: The neural network model used for leaf evaluation and policy priors.
            use_policy_head (bool): Whether to use the policy head of the neural network.
        """
        # Determines if the NN's policy output guides the search.
        self.use_policy_head = use_policy_head
        # Exploration constant in the UCT formula. Balances exploitation and exploration.
        self.cpuct = 0.1
        # Epsilon factor for mixing Dirichlet noise into root priors (used during training).
        self.e = 0.75
        self.nnet_model = nnet_model
        
        # Get the device for the neural network
        self.device = next(nnet_model.parameters()).device

        # MCTS Tree storage: Dictionaries keyed by state (bytes representation).
        self.Qs = {}  # Stores Q-values (expected reward) for states.
        # Stores lower bounds on state values (TEMCTS feature).
        self.v_lower_bound = {}
        # Stores upper bounds on state values (TEMCTS feature).
        self.v_upper_bound = {}
        # Stores visit counts for (state, action) pairs.
        self.Nsa = {}
        # Stores visit counts for states.
        self.Ns = {}
        # Stores game end results for terminal states. (Seems unused?)
        self.Es = {}
        # Stores prior probabilities for actions in states (from NN policy head).
        self.Ps = {}

        # Search control parameters
        self.mcts_sims = 0 # Counter for simulations in the current search.
        self.max_sims = None # Optional hard limit on the number of simulations.
        # KL-divergence threshold for dynamic search termination.
        self.kld_threshold = None
        # Current KL-divergence between recent move probability distributions.
        self.kl_divergence = None
        # Dirichlet noise vector added to root priors during training.
        self.noise = None
        # Batch size for NN predictions if batching were implemented. (Seems unused?)
        self.eval_batch_size = 5
        self.max_depth = 0 # Tracks the maximum depth reached during search.

    def get_probabilities(self, board, player, kld_threshold, max_sims=None, temperature=1, verbose=False, dir_alpha=0):
        """
        Runs the MCTS search from the given board state to generate move probabilities.

        Args:
            board: The current board state (checkersBoard.CheckersBoard object).
            player: The player whose turn it is.
            kld_threshold: KL-divergence threshold for terminating the search dynamically.
            max_sims (int, optional): Maximum number of simulations to run. Defaults to None.
            temperature (float): Controls the exploration/exploitation balance in the final move selection.
                                 0 means deterministic selection (pick best), >0 means probabilistic.
            verbose (bool): If True, print search statistics during the process.
            dir_alpha (float): Alpha parameter for Dirichlet noise added to root priors.
                               Set > 0 during training self-play for exploration.

        Returns:
            tuple: (exponentiated_probs, node_probs)
                   - exponentiated_probs: Final move probabilities, adjusted by temperature.
                   - node_probs: Raw move probabilities based on visit counts before temperature application.
        """
        self.mcts_sims = 0
        self.max_sims = max_sims
        node_probs = None # Stores the probability distribution from the previous check.
        self.kl_divergence = math.inf
        self.kld_threshold = kld_threshold
        # Use byte representation of features as the state key for dictionaries.
        state = TDAgent.extract_features(board, player).tobytes()
        valid_moves = board.get_valid_moves(player, include_index=True, include_chain_jumps=False)

        # Add Dirichlet noise to root priors for exploration during training.
        if dir_alpha > 0:
            # Ensure noise array length matches the number of valid moves
            self.noise = np.random.dirichlet([dir_alpha] * len(valid_moves))
        else:
            self.noise = None

        # Initialize bounds for the root state.
        self.v_lower_bound[(state)] = -math.inf
        self.v_upper_bound[(state)] = math.inf
        self.max_depth = 0

        # Main MCTS loop: continues until termination condition is met.
        while not self.terminate_search():
            self.mcts_sims += 1
            # Perform one MCTS simulation (select, expand, backup).
            self.search(board, self.v_lower_bound[(state)], self.v_upper_bound[(state)], 0, dir_alpha)

            # Periodically check for KL-divergence convergence.
            if self.mcts_sims % 25 == 0:
                # print(f"[MCTS Debug] Check KL divergence at sim {self.mcts_sims}") # DEBUG
                new_probs = np.zeros(checkersBoard.CheckersBoard.action_size)
                counts = np.zeros(checkersBoard.CheckersBoard.action_size)
                q_values = np.zeros(checkersBoard.CheckersBoard.action_size)
                lower_bounds = []
                upper_bounds = []
                # Calculate current move probabilities based on visit counts.
                for move, index in valid_moves:
                    index = int(index)
                    # Key for action is the state resulting from the move.
                    move_state = TDAgent.extract_features(move, move.current_player).tobytes()
                    counts[index] = self.Nsa.get((state, move_state), 0) # Use .get for safety
                    q_values[index] = self.Qs.get(move_state, 0) # Q value of the *resulting* state
                    lower_bounds.append(self.v_lower_bound.get(move_state, -math.inf))
                    upper_bounds.append(self.v_upper_bound.get(move_state, math.inf))

                if sum(counts) > 0: # Avoid division by zero if no visits yet
                    new_probs = [x/sum(counts) for x in counts]
                    # Calculate KL divergence if we have a previous distribution.
                    if node_probs is not None:
                        # print(f"[MCTS Debug] Calculating KL divergence. Prev Probs Nonzero: {np.count_nonzero(node_probs)}, New Probs Nonzero: {np.count_nonzero(new_probs)}") # DEBUG
                        # print(f"[MCTS Debug] Prev Probs: {node_probs}") # DEBUG
                        # print(f"[MCTS Debug] New Probs: {new_probs}") # DEBUG
                        try:
                            # Ensure inputs are valid for entropy (non-negative, sum to 1 ideally)
                            # Scipy usually handles near-zero values, but let's be cautious
                            clean_new_probs = np.maximum(new_probs, 0) # Ensure non-negative
                            clean_node_probs = np.maximum(node_probs, 0)
                            sum_new = np.sum(clean_new_probs)
                            sum_old = np.sum(clean_node_probs)
                            if sum_new > 1e-6 and sum_old > 1e-6: # Only calculate if sums are valid
                                 clean_new_probs /= sum_new
                                 clean_node_probs /= sum_old
                                 self.kl_divergence = stats.entropy(clean_new_probs, clean_node_probs)
                                 # print(f"[MCTS Debug] Calculated KL: {self.kl_divergence}") # DEBUG
                            else:
                                 self.kl_divergence = math.inf # Assign inf if probs are invalid
                                 # print(f"[MCTS Debug] Invalid probs for KL calc (sums: new={sum_new}, old={sum_old}), setting KL=inf") # DEBUG
                        except Exception as e:
                            print(f"[MCTS Error] Exception during KL divergence calculation: {e}")
                            print(f"[MCTS Error] new_probs: {new_probs}")
                            print(f"[MCTS Error] node_probs: {node_probs}")
                            self.kl_divergence = math.inf # Assign inf on error
                    # Update the stored probability distribution.
                    node_probs = new_probs

                if verbose:
                    root_q = self.Qs.get(state, "N/A")
                    root_lower = self.v_lower_bound.get(state, "N/A")
                    root_upper = self.v_upper_bound.get(state, "N/A")
                    print('root q: ' + str(root_q) + ' v-: ' + str(root_lower) + ' v+: ' + str(root_upper))
                    print('counts: ' + str([x for x in counts if x != 0]))
                    print('probabilities: ' + str([x for x in node_probs if x != 0 and node_probs is not None]))
                    print('q values (children): ' + str([x for x in q_values if x != 0]))
                    print('lower bounds (children): ' + str(lower_bounds))
                    print('upper bounds (children): ' + str(upper_bounds))
                    print(f'KL Divergence: {self.kl_divergence}, Threshold: {self.kld_threshold}, Sims: {self.mcts_sims}')


        if verbose:
             if state in self.Ps: print('prior probs: ' + str([x for x in self.Ps[state] if x != 0]))
             else: print('prior probs: Not calculated (no expansion needed?)')


        # Prepare final move probabilities based on visit counts and temperature.
        counts = np.zeros(checkersBoard.CheckersBoard.action_size)
        child_scores = np.zeros(checkersBoard.CheckersBoard.action_size) # Use Q-values to break ties if temp=0
        for move, index in valid_moves:
            move_state = TDAgent.extract_features(move, move.current_player).tobytes()
            counts[int(index)] = self.Nsa.get((state, move_state), 0)
            # Use Q value of child state as the score for deterministic selection
            child_scores[int(index)] = self.Qs.get(move_state, -2) # Default to low score if not visited

        if temperature == 0:
            # Deterministic: choose the move with the highest visit count (break ties with Q-value).
            # Find indices of max counts
            max_count = np.max(counts)
            best_indices = np.where(counts == max_count)[0]
            # If multiple moves have max count, use Q-value to break ties
            if len(best_indices) > 1:
                 best_move_index = best_indices[np.argmax(child_scores[best_indices])]
            elif len(best_indices) == 1:
                 best_move_index = best_indices[0]
            else: # Should not happen if valid_moves is not empty
                 # If no visits, pick based on highest child score (Q-value estimate)
                 best_move_index = np.argmax(child_scores)

            exponentiated_probs = np.zeros_like(counts)
            if len(valid_moves) > 0: # Ensure there's at least one move
                 exponentiated_probs[best_move_index] = 1
        else:
            # Probabilistic: sample move based on visit counts raised to 1/temperature.
            counts_temp = [x**(1/temperature) for x in counts]
            sum_counts_temp = float(sum(counts_temp))
            if sum_counts_temp == 0: # Handle case where root might not have been expanded or all counts zero
                 # Return uniform probability over valid moves if no simulations ran or counts are zero
                 num_valid = len(valid_moves)
                 uniform_prob = 1.0 / num_valid if num_valid > 0 else 0
                 exponentiated_probs = np.zeros(checkersBoard.CheckersBoard.action_size)
                 for _, index in valid_moves:
                      exponentiated_probs[int(index)] = uniform_prob
            else:
                 exponentiated_probs = [x/sum_counts_temp if x != 0 else 0 for x in counts_temp]

        return exponentiated_probs, node_probs

    # def batch_search(self, root_node, dir_alpha=0):


    def search(self, board, alpha, beta, depth, dir_alpha=0):
        """
        Performs one simulation step (selection, expansion, backup).

        Args:
            board: The current board state.
            alpha: Lower bound for the current search path (TEMCTS feature).
            beta: Upper bound for the current search path (TEMCTS feature).
            depth: The current depth in the search tree.
            dir_alpha: Dirichlet noise parameter (only used at root, passed down).

        Returns:
            None: Values are updated in the tree structure during backup.
        """
        self.max_depth = max(self.max_depth, depth)
        depth = depth + 1

        # print(f"[MCTS Debug] Enter search: depth={depth}, player={board.current_player}, state={TDAgent.extract_features(board, board.current_player).tobytes()[:10]}...") # DEBUG

        current_player = board.current_player
        state = TDAgent.extract_features(board, current_player).tobytes()
        if state not in self.Ns:
            self.Ns[state] = 0

        # Check if the game has ended at this node.
        ended, winner = board.game_ended()
        if ended:
            # Terminal node: Backup the exact game result.
            # Value is stored from Player 1's perspective (+1, -1, or 0).
            self.Qs[state] = winner
            self.v_lower_bound[state] = winner
            self.v_upper_bound[state] = winner
            # In standard MCTS, we'd return the value here for backup.
            # This implementation does backup differently (modifying Qs later).
            return # Return winner?

        valid_moves = board.get_valid_moves(current_player, include_index=True, include_chain_jumps=False)

        # --- Expansion Phase ---: If the node hasn't been visited before.
        if self.Ns[state] == 0:
            # Initialize children nodes' basic info.
            valids = np.zeros(checkersBoard.CheckersBoard.action_size)
            for move, idx in valid_moves:
                new_state = TDAgent.extract_features(move, move.current_player).tobytes()
                if new_state not in self.Ns: # Initialize only if truly new
                    self.Ns[new_state] = 0
                    self.v_lower_bound[new_state] = -math.inf
                    self.v_upper_bound[new_state] = math.inf
                valids[int(idx)] = 1 # Mark index as valid move

            self.Ns[state] += 1 # Mark current node as visited once (for expansion).

            # Re-check if the *current* board state leads to immediate end (should be rare)
            ended, winner = board.game_ended()
            if ended:
                self.Qs[state] = winner
                self.v_lower_bound[state] = winner
                self.v_upper_bound[state] = winner
                return # Return winner?

            # Evaluate the current board state using the neural network.
            features = TDAgent.extract_features(board, current_player)
            features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            
            # print(f"[MCTS Debug] Expanding node at depth {depth}. Calling NN predict.") # DEBUG
            nn_output = None
            try:
                # Check for valid features shape before prediction
                if features.shape != (5, checkersBoard.CheckersBoard.board_height, 
                                     checkersBoard.CheckersBoard.board_width):
                    print(f"[MCTS Error] Invalid features shape: {features.shape}. Expected (5, 8, 8).")
                    raise ValueError("Invalid features shape")
                    
                # Handle potential NaN values in features
                if np.isnan(features).any():
                    print(f"[MCTS Error] NaN values in features.")
                    features = np.nan_to_num(features, nan=0.0)
                    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
                
                # Set model to evaluation mode and run inference
                self.nnet_model.eval()
                with torch.no_grad():
                    nn_output = self.nnet_model(features_tensor)
                # print(f"[MCTS Debug] NN predict output: {nn_output}") # DEBUG
            except Exception as e:
                print(f"[MCTS Error] Exception during NN prediction: {e}")
                import traceback
                traceback.print_exc()
                # Decide how to handle NN error - maybe return neutral value?
                v = 0.0
                pi = np.ones_like(valids) * valids # Uniform policy over valid moves
                sum_pi = np.sum(pi)
                if sum_pi > 0: pi /= sum_pi
                self.Ps[state] = pi
                self.Qs[state] = v
                self.v_lower_bound[state] = v
                self.v_upper_bound[state] = v
                return

            if self.use_policy_head:
                # Handle PyTorch model output format which returns a tuple (value, policy)
                if isinstance(nn_output, tuple):
                    v, pi = nn_output
                    # Extract from tensors to numpy
                    v = v.squeeze().cpu().numpy()
                    pi = pi.squeeze().cpu().numpy()
                else:
                    # If only value is returned
                    v = nn_output.squeeze().cpu().numpy()
                    pi = np.zeros(checkersBoard.CheckersBoard.action_size)  # Default to uniform later

                if sum(valids) == 0:
                    # Should not happen if game_ended() is correct.
                    print('Error: No valid moves found during expansion for player {current_player}.')
                    # Assign loss value and zero policy.
                    v = -1.0 if current_player == 1 else 1.0 # Loss for current player (P1 perspective)
                    pi = np.zeros_like(pi)
                    self.Ps[state] = pi
                else:
                    pi *= valids # Mask policy with valid moves.
                    sum_pi = np.sum(pi)
                    if sum_pi > 1e-6: # Normalize if sum is non-negligible
                        pi /= sum_pi
                    else:
                        # If NN gives zero prior to all valid moves, use uniform.
                        print(f"Warning: Zero policy priors for player {current_player}. Using uniform.")
                        num_valid = int(np.sum(valids))
                        pi = valids / num_valid if num_valid > 0 else valids # Avoid div by zero
                    self.Ps[state] = pi # Store the policy priors.
            else:
                # If not using policy head, only get the value prediction.
                if isinstance(nn_output, tuple):
                    v = nn_output[0].squeeze().cpu().numpy()  # Extract value from tuple
                else:
                    v = nn_output.squeeze().cpu().numpy()  # Only value was returned
                
                # Use uniform priors implicitly during selection if Ps[state] is missing.
                num_valid = int(np.sum(valids))
                uniform_prob = 1.0 / num_valid if num_valid > 0 else 0
                self.Ps[state] = np.array([uniform_prob if valids[i] else 0 for i in range(checkersBoard.CheckersBoard.action_size)])

            # Initialize Q value and bounds with the NN's evaluation.
            # Store value from Player 1's perspective.
            self.Qs[state] = v
            self.v_lower_bound[state] = v
            self.v_upper_bound[state] = v
            # Return value for backup? Not needed in this backup style.
            return

        # --- Selection Phase --- : Node has been visited, select best child.
        # Find the best action (move) to take based on UCT formula.
        best_u = -math.inf
        best_move = None
        best_move_state = None
        i = 0 # Index for applying Dirichlet noise if needed.

        # Calculate UCT scores for all valid moves.
        for move, move_index in valid_moves:
            try:
                move_index = int(move_index)
                move_state = TDAgent.extract_features(move, move.current_player).tobytes()

                # Get Q-value and visit counts for the action (state -> move_state)
                action_visits = self.Nsa.get((state, move_state), 0)
                child_q = self.Qs.get(move_state, 0) # Default Q=0 if child not expanded yet
                
                # Check for NaN values
                if math.isnan(child_q):
                    print(f"[MCTS Warning] Found NaN Q-value for move {move_index}. Setting to 0.")
                    child_q = 0

                # Prior probability for this action
                prior_p = 0
                if state in self.Ps and len(self.Ps[state]) > move_index:
                    prior_p = self.Ps[state][move_index]
                    if math.isnan(prior_p):
                        print(f"[MCTS Warning] Found NaN prior for move {move_index}. Setting to small value.")
                        prior_p = 0.001

                # Apply Dirichlet noise at the root node during training
                if depth == 1 and dir_alpha > 0 and self.use_policy_head and self.noise is not None and i < len(self.noise):
                    prior_p = (prior_p * (1 - self.e)) + self.noise[i] * self.e

                # Calculate UCT score
                if action_visits > 0:
                    # Value from current player's perspective
                    q_value_perspective = child_q * current_player
                    uct_score = q_value_perspective + self.cpuct * prior_p * (math.sqrt(self.Ns[state]) / (1 + action_visits))
                else:
                    # If action not taken, use prior probability and parent visit count
                    # Give a bonus to unexplored actions based on their prior
                    # (Simplified UCT term for unvisited actions)
                    uct_score = self.cpuct * prior_p * math.sqrt(self.Ns[state] + 1e-8) # Add epsilon for sqrt(0)

                if uct_score > best_u:
                    best_u = uct_score
                    best_move = move
                    best_move_state = move_state
            except Exception as e:
                print(f"[MCTS Error] Exception during move evaluation: {e}")
            i += 1

        # --- Recursive Call --- : Recurse on the selected best move.
        if best_move is not None:
            # Pass the alpha-beta bounds associated with the chosen child state.
            # Note: The effectiveness of passing these bounds depends on how they are used/updated.
            # child_alpha = self.v_lower_bound.get(best_move_state, -math.inf)
            # child_beta = self.v_upper_bound.get(best_move_state, math.inf)
            # Simplified call without passing bounds for now, as their update/use needs review.
            self.search(best_move, alpha, beta, depth) # Pass original alpha/beta down? Or child bounds?
        else:
             # This should only happen if valid_moves was empty, which game_ended should catch.
             print(f"Error: No best move found for player {current_player} at depth {depth}. Valid moves: {len(valid_moves)}")
             return # Cannot proceed


        # --- Backpropagation Phase --- : Update stats after recursion returns.
        # Update visit counts for the chosen state-action edge and the parent state.
        if best_move_state is not None:
             self.Nsa[(state, best_move_state)] = self.Nsa.get((state, best_move_state), 0) + 1

        # Update Q-value and Bounds for the current state (state) based on children's current values/bounds.
        # This update logic is non-standard for MCTS. It's a minimax-like update.
        # It takes the max/min of the Q-values of *all currently known* children.
        child_q_values = []
        child_lower_bounds = [] # Collect bounds for potential update
        all_children_expanded = True # Assume all are expanded initially
        for move, move_index in valid_moves:
            move_state_i = TDAgent.extract_features(move, move.current_player).tobytes()
            if move_state_i in self.Qs:
                 child_q_values.append(self.Qs[move_state_i])
                 child_lower_bounds.append(self.v_lower_bound.get(move_state_i, -math.inf))
                 # child_upper_bounds.append(self.v_upper_bound.get(move_state_i, math.inf))
            else:
                 all_children_expanded = False # Mark if any child is not yet expanded

        if child_q_values: # If at least one child has been evaluated
             if current_player == 1: # Maximizing player (wants high Q-value)
                  new_q = max(child_q_values)
                  # Update Q if all children are expanded (perfect info) or if new max is better than current Q.
                  if all_children_expanded or new_q > self.Qs.get(state, -math.inf):
                       self.Qs[state] = new_q
                  # Update lower bound based on the max lower bound achievable through children.
                  if child_lower_bounds: self.v_lower_bound[state] = max(child_lower_bounds)
                  # Upper bound update is less clear in minimax context from children.
             else: # Minimizing player (-1) (wants low Q-value)
                  new_q = min(child_q_values)
                  # Update Q if all children are expanded or if new min is better than current Q.
                  if all_children_expanded or new_q < self.Qs.get(state, math.inf):
                       self.Qs[state] = new_q
                  # Update lower bound based on the min lower bound achievable through children (from P1 perspective).
                  if child_lower_bounds: self.v_lower_bound[state] = min(child_lower_bounds)
                  # Upper bound update is less clear.

        # Increment visit count for the current state AFTER the recursive call and updates.
        self.Ns[state] += 1
        # print(f"[MCTS Debug] Exit search: depth={depth}") # DEBUG
        return

    def terminate_search(self):
        """
        Checks if the MCTS search should terminate based on configured criteria.

        Returns:
            bool: True if the search should terminate, False otherwise.
        """
        # Option 1: Terminate after a fixed number of simulations.
        if self.max_sims is not None:
            if self.max_sims <= self.mcts_sims:
                return True
            # else: continue to check KL divergence if max_sims not reached

        # Option 2: Terminate if KL divergence between probability distributions stabilizes.
        # Requires self.kl_divergence to be calculated and valid (not inf).
        if self.kl_divergence is not None and self.kl_divergence != math.inf:
             if self.kld_threshold >= self.kl_divergence:
                 # print(f"Terminating search due to KL divergence {self.kl_divergence} <= {self.kld_threshold} at sim {self.mcts_sims}")
                 return True

        # Continue search if neither condition met.
        return False
