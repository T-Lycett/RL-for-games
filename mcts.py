import TDAgent
import math
import numpy as np
import torch
import checkersBoard
from scipy import stats


class MCTS:
    """
    Implements a Monte Carlo Tree Search algorithm enhanced with neural network guidance,
    following the AlphaZero methodology more closely for backup.
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
        self.cpuct = 1.0 # Increased cpuct typical for AZ style
        # Epsilon factor for mixing Dirichlet noise into root priors (used during training).
        self.e = 0.25 # Standard AZ value
        self.nnet_model = nnet_model

        # Get the device for the neural network
        self.device = next(nnet_model.parameters()).device

        # MCTS Tree storage: Dictionaries keyed by state (bytes representation).
        # Qs[state][action_state] stores Q-values (expected reward) for taking action leading to action_state from state.
        self.Qs = {}
        # Nsa[state][action_state] stores visit counts for the edge (state, action_state).
        self.Nsa = {}
        # Ns[state] stores total visit counts for state s (sum of Nsa[s][a] over all a).
        self.Ns = {}
        # Es stores game end results cached for terminal states. -1=P2 win, 0=draw, 1=P1 win
        self.Es = {}
        # Ps stores prior probabilities P(a|s) for actions in states (from NN policy head).
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
            num_valid_moves = len(valid_moves)
            if num_valid_moves > 0: # Avoid error if no valid moves
                self.noise = np.random.dirichlet([dir_alpha] * num_valid_moves)
            else:
                self.noise = None # No noise if no moves
        else:
            self.noise = None

        self.max_depth = 0

        # Main MCTS loop: continues until termination condition is met.
        while not self.terminate_search():
            self.mcts_sims += 1
            # Perform one MCTS simulation (select, expand, backup).
            # search returns the value from the perspective of the player at the board state
            self.search(board, 0, dir_alpha)

            # Periodically check for KL-divergence convergence.
            if self.mcts_sims % 40 == 0:
                # print(f"[MCTS Debug] Check KL divergence at sim {self.mcts_sims}") # DEBUG
                new_probs = np.zeros(checkersBoard.CheckersBoard.action_size)
                counts = np.zeros(checkersBoard.CheckersBoard.action_size)
                q_values = np.zeros(checkersBoard.CheckersBoard.action_size)
                # Calculate current move probabilities based on visit counts.
                for move, index in valid_moves:
                    index = int(index)
                    # Key for action is the state resulting from the move.
                    move_state = TDAgent.extract_features(move, move.current_player).tobytes()
                    # Use Nsa to get action counts
                    counts[index] = self.Nsa.get(state, {}).get(move_state, 0)
                    # Use Qs to get action Q-values
                    q_values[index] = self.Qs.get(state, {}).get(move_state, 0)

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
                            clean_new_probs = np.maximum(new_probs, 1e-9) # Add epsilon for stability
                            clean_node_probs = np.maximum(node_probs, 1e-9)
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
                    # Get root node's average Q value (across actions) for display? Or just Ns?
                    root_visits = self.Ns.get(state, 0)
                    print(f'Root Visits: {root_visits}')
                    print('counts: ' + str([x for x in counts if x != 0]))
                    print('probabilities: ' + str([x for x in node_probs if x != 0 and node_probs is not None]))
                    print('q values (actions): ' + str([q for q, c in zip(q_values, counts) if c != 0])) # Show Q-values for visited actions
                    print(f'KL Divergence: {self.kl_divergence}, Threshold: {self.kld_threshold}, Sims: {self.mcts_sims}')


        if verbose:
             if state in self.Ps: print('prior probs: ' + str([x for x in self.Ps[state] if x != 0]))
             else: print('prior probs: Not calculated (no expansion needed?)')


        # Prepare final move probabilities based on visit counts and temperature.
        counts = np.zeros(checkersBoard.CheckersBoard.action_size)
        child_scores = np.zeros(checkersBoard.CheckersBoard.action_size) # Use Q-values to break ties if temp=0
        for move, index in valid_moves:
            index = int(index)
            move_state = TDAgent.extract_features(move, move.current_player).tobytes()
            counts[index] = self.Nsa.get(state, {}).get(move_state, 0)
            # Use action Q value as the score for deterministic selection
            child_scores[index] = self.Qs.get(state, {}).get(move_state, -2) # Default to low score if not visited

        if temperature == 0:
            # Deterministic: choose the move with the highest visit count (break ties with Q-value).
            # Find indices of max counts
            max_count = np.max(counts)
            # Check if any moves were visited
            if max_count == 0:
                 # If no visits (e.g., max_sims=0 or immediate termination), pick based on raw Q-value (child_scores)
                 if len(valid_moves) > 0:
                      best_move_index = np.argmax(child_scores) # Scores already fetched
                 else:
                      best_move_index = -1 # Should not happen
            else:
                 best_indices = np.where(counts == max_count)[0]
                 # If multiple moves have max count, use Q-value to break ties
                 if len(best_indices) > 1:
                      best_move_index = best_indices[np.argmax(child_scores[best_indices])]
                 elif len(best_indices) == 1:
                      best_move_index = best_indices[0]
                 else: # Should not happen if valid_moves is not empty and max_count > 0
                      print(f"[MCTS Warning] No best index found with max_count={max_count}. Fallback to argmax child_scores.")
                      best_move_index = np.argmax(child_scores)

            exponentiated_probs = np.zeros_like(counts)
            if len(valid_moves) > 0 and best_move_index != -1: # Ensure there's at least one move and a valid index
                 exponentiated_probs[best_move_index] = 1
            elif len(valid_moves) > 0: # Fallback if best_move_index invalid but moves exist
                 print("[MCTS Warning] Invalid best_move_index with temp=0. Using first valid move.")
                 exponentiated_probs[int(valid_moves[0][1])] = 1

        else:
            # Probabilistic: sample move based on visit counts raised to 1/temperature.
            counts_temp = [x**(1/temperature) for x in counts]
            sum_counts_temp = float(sum(counts_temp))
            if sum_counts_temp < 1e-9: # Handle case where root might not have been expanded or all counts zero
                 # Return uniform probability over valid moves if no simulations ran or counts are zero
                 num_valid = len(valid_moves)
                 uniform_prob = 1.0 / num_valid if num_valid > 0 else 0
                 exponentiated_probs = np.zeros(checkersBoard.CheckersBoard.action_size)
                 for _, index in valid_moves:
                      exponentiated_probs[int(index)] = uniform_prob
            else:
                 exponentiated_probs = [(x/sum_counts_temp) if x > 1e-9 else 0 for x in counts_temp] # Avoid NaNs

        # Final safety check for probability sum
        prob_sum = np.sum(exponentiated_probs)
        if abs(prob_sum - 1.0) > 0.01 and prob_sum > 1e-9 :
            print(f"[MCTS Warning] Probs sum to {prob_sum}. Renormalizing.")
            exponentiated_probs = exponentiated_probs / prob_sum
        elif prob_sum < 1e-9 and len(valid_moves) > 0:
            print(f"[MCTS Warning] Probs sum to {prob_sum} with valid moves. Setting uniform.")
            num_valid = len(valid_moves)
            uniform_prob = 1.0 / num_valid
            exponentiated_probs = np.zeros(checkersBoard.CheckersBoard.action_size)
            for _, index in valid_moves:
                 exponentiated_probs[int(index)] = uniform_prob


        return exponentiated_probs, node_probs

    # def batch_search(self, root_node, dir_alpha=0):


    def search(self, board, depth, dir_alpha=0):
        """
        Performs one simulation step (selection, expansion, backup) using AlphaZero logic.

        Args:
            board: The current board state.
            depth: The current depth in the search tree.
            dir_alpha: Dirichlet noise parameter (only used at root).

        Returns:
            float: The value of the current state from the perspective of board.current_player.
                   (-v from the perspective of the parent node's player).
        """
        self.max_depth = max(self.max_depth, depth)
        current_player = board.current_player
        state = TDAgent.extract_features(board, current_player).tobytes()

        # --- Terminal Node Check ---
        if state in self.Es: # Check cache first
            return self.Es[state]
        ended, winner = board.game_ended()
        if ended:
            # Cache and return the exact game result (-1, 0, or 1 from P1 perspective).
            # The return value should be from the perspective of the current_player.
            value_for_current_player = winner if current_player == 1 else -winner
            self.Es[state] = value_for_current_player
            return value_for_current_player

        valid_moves = board.get_valid_moves(current_player, include_index=True, include_chain_jumps=False)

        # --- Expansion Phase ---: If the node hasn't been evaluated by NN (check Ps).
        if state not in self.Ps:
            # Evaluate the current board state using the neural network.
            features = TDAgent.extract_features(board, current_player)
            features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)

            # print(f"[MCTS Debug] Expanding node at depth {depth}. Calling NN predict.") # DEBUG
            nn_output = None
            v = 0.0 # Default value on error
            pi = None # Default policy on error
            try:
                # Check for valid features shape before prediction
                if features.shape != (5, checkersBoard.CheckersBoard.board_height,
                                     checkersBoard.CheckersBoard.board_width):
                    print(f"[MCTS Error] Invalid features shape: {features.shape}. Expected (5, 8, 8). State: {state[:20]}...")
                    # Fallback: uniform policy, neutral value
                    valids_mask = np.zeros(checkersBoard.CheckersBoard.action_size)
                    for _, idx in valid_moves: valids_mask[int(idx)] = 1
                    num_valid = len(valid_moves)
                    pi = valids_mask / num_valid if num_valid > 0 else valids_mask
                    v = 0.0

                else:
                    # Handle potential NaN values in features
                    if np.isnan(features).any():
                        print(f"[MCTS Warning] NaN values in features detected at depth {depth}. Replacing with 0.")
                        features = np.nan_to_num(features, nan=0.0)
                        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)

                    # Set model to evaluation mode and run inference
                    self.nnet_model.eval()
                    with torch.no_grad():
                        nn_output = self.nnet_model(features_tensor)
                    # print(f"[MCTS Debug] NN predict output: {nn_output}") # DEBUG

                    # --- Process NN Output ---
                    if self.use_policy_head:
                        if isinstance(nn_output, tuple):
                            v_tensor, pi_tensor = nn_output
                            v = v_tensor.squeeze().cpu().item() # Use .item() for scalar
                            pi = pi_tensor.squeeze().cpu().numpy()
                        else: # Only value head used/returned
                            v = nn_output.squeeze().cpu().item()
                            pi = np.ones(checkersBoard.CheckersBoard.action_size) # Placeholder for uniform policy generation
                    else: # Not using policy head (e.g., pure Q-learning MCTS)
                        if isinstance(nn_output, tuple):
                            v = nn_output[0].squeeze().cpu().item()
                        else:
                            v = nn_output.squeeze().cpu().item()
                        pi = np.ones(checkersBoard.CheckersBoard.action_size) # Placeholder for uniform policy generation

                    # --- Apply Softmax to policy logits ---
                    # Convert raw logits (pi) to probabilities using Softmax
                    # Do this *before* masking
                    if pi is not None and self.use_policy_head: # Check if pi exists and policy head is used
                         pi_tensor = torch.from_numpy(pi).float() # Convert back to tensor if needed
                         pi = torch.softmax(pi_tensor, dim=0).numpy() # Apply softmax

                    # --- Validate and Normalize Policy ---
                    valids_mask = np.zeros(checkersBoard.CheckersBoard.action_size)
                    for _, idx in valid_moves: valids_mask[int(idx)] = 1

                    if sum(valids_mask) == 0 and not ended:
                        print(f'Error: No valid moves found during expansion for player {current_player}, but game not ended.')
                        # Assign loss value and zero policy?
                        v = -1.0 # Loss for current player
                        pi = np.zeros_like(pi)
                    elif not self.use_policy_head:
                         # If not using policy head, create uniform priors over valid moves
                         num_valid = len(valid_moves)
                         pi = valids_mask / num_valid if num_valid > 0 else valids_mask
                    else:
                        pi = pi * valids_mask # Mask policy with valid moves.
                        sum_pi = np.sum(pi)
                        # --- DEBUG PRINTS ---
                        if depth == 0: # Only print for the root node expansion
                            print(f"[MCTS Debug Expansion D{depth}] Raw NN pi (masked): {pi[pi>0]}")
                            print(f"[MCTS Debug Expansion D{depth}] Sum of masked pi: {sum_pi}")
                        # --- END DEBUG PRINTS ---
                        if sum_pi > 1e-6: # Normalize if sum is non-negligible
                            pi /= sum_pi
                        else:
                            # If NN gives zero prior to all valid moves, use uniform.
                            print(f"Warning: Zero policy priors for player {current_player} at depth {depth}. Using uniform.")
                            num_valid = len(valid_moves)
                            pi = valids_mask / num_valid if num_valid > 0 else valids_mask # Avoid div by zero

            except Exception as e:
                print(f"[MCTS Error] Exception during NN prediction or processing: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: uniform policy, neutral value
                valids_mask = np.zeros(checkersBoard.CheckersBoard.action_size)
                for _, idx in valid_moves: valids_mask[int(idx)] = 1
                num_valid = len(valid_moves)
                pi = valids_mask / num_valid if num_valid > 0 else valids_mask
                v = 0.0


            # --- Store Priors and Initialize Counts ---
            if pi is None: # Ensure pi is initialized even if errors occurred
                 print(f"[MCTS Error] Policy pi is None after NN evaluation for player {current_player} at depth {depth}. Using uniform fallback.")
                 valids_mask = np.zeros(checkersBoard.CheckersBoard.action_size)
                 for _, idx in valid_moves: valids_mask[int(idx)] = 1
                 num_valid = len(valid_moves)
                 pi = valids_mask / num_valid if num_valid > 0 else valids_mask

            self.Ps[state] = pi
            self.Ns[state] = 0 # Visit count initialized to 0, will be incremented during backup.
            # Return the NN's value estimate v. Backup happens in parent call.
            # Value v is from the perspective of the current_player.
            return v


        # --- Selection Phase --- : Node has been visited, select best child using PUCT.
        best_u = -math.inf
        best_move = None
        best_move_state = None
        noise_idx = 0 # Index for applying Dirichlet noise if needed.

        # Calculate PUCT scores for all valid moves.
        for move, move_index in valid_moves:
            try:
                move_index = int(move_index)
                move_state = TDAgent.extract_features(move, move.current_player).tobytes()

                # Get Q-value and visit counts for the action (state -> move_state)
                # Default Q=0, N=0 if edge not explored yet
                q_value = self.Qs.get(state, {}).get(move_state, 0)
                action_visits = self.Nsa.get(state, {}).get(move_state, 0)

                # Prior probability for this action
                prior_p = 0
                # Add safeguard for Ps not having state (should not happen after expansion logic)
                if state in self.Ps and len(self.Ps[state]) > move_index:
                    prior_p = self.Ps[state][move_index]
                    if math.isnan(prior_p):
                        print(f"[MCTS Warning] Found NaN prior for move {move_index}. Setting to small value.")
                        prior_p = 1e-6 # Use small positive value

                # Apply Dirichlet noise at the root node during training self-play
                if depth == 0 and dir_alpha > 0 and self.noise is not None and noise_idx < len(self.noise):
                    prior_p = (prior_p * (1 - self.e)) + self.noise[noise_idx] * self.e
                    noise_idx += 1 # Increment noise index only when used

                # Calculate PUCT score
                # Q-value is already from the perspective of the current player selecting the action.
                # Parent visit count Ns[state] is needed. Add epsilon for sqrt robustness.
                parent_visits = self.Ns.get(state, 0)
                uct_score = q_value + self.cpuct * prior_p * (math.sqrt(parent_visits + 1e-8) / (1 + action_visits))

                if uct_score > best_u:
                    best_u = uct_score
                    best_move = move
                    best_move_state = move_state
            except Exception as e:
                print(f"[MCTS Error] Exception during PUCT calculation for move {move_index}: {e}")
                import traceback
                traceback.print_exc()


        # --- Recursive Call --- : Recurse on the selected best move.
        if best_move is not None and best_move_state is not None:
            # Recursive call returns value v from the perspective of the player in best_move state
            v = self.search(best_move, depth + 1, dir_alpha) # No need to pass dir_alpha down further
        else:
             # This should only happen if valid_moves was empty (caught by terminal check)
             # or if an error occurred during selection.
             print(f"Error: No best move found for player {current_player} at depth {depth}. Valid moves: {len(valid_moves)}")
             # Fallback: return neutral value?
             return 0.0 # Return neutral value if selection failed


        # --- Backpropagation Phase (AlphaZero Style) ---
        # Update stats for the edge (state, best_move_state) using the returned value v.
        # v is the value from the perspective of the player in best_move (the opponent).
        # We need to use -v for the current player's update.
        value_for_parent = -v

        # Initialize dictionaries if this edge hasn't been visited before
        if state not in self.Qs: self.Qs[state] = {}
        if state not in self.Nsa: self.Nsa[state] = {}

        # Update Q-value for the action: Q(s,a) = (N(s,a)*Q(s,a) + v) / (N(s,a)+1)
        old_q = self.Qs[state].get(best_move_state, 0)
        old_n = self.Nsa[state].get(best_move_state, 0)
        self.Qs[state][best_move_state] = (old_n * old_q + value_for_parent) / (old_n + 1)

        # Update action visit count N(s,a)
        self.Nsa[state][best_move_state] = old_n + 1

        # Update parent state visit count N(s)
        self.Ns[state] = self.Ns.get(state, 0) + 1

        # Return the value from the perspective of the current player at 'state'
        return value_for_parent


    def terminate_search(self):
        """
        Checks if the MCTS search should terminate based on configured criteria.

        Returns:
            bool: True if the search should terminate, False otherwise.
        """
        # Option 1: Terminate after a fixed number of simulations.
        if self.max_sims is not None:
            if self.max_sims <= self.mcts_sims:
                # print(f"Terminating search due to max_sims {self.max_sims} reached at sim {self.mcts_sims}")
                return True
            # else: continue to check KL divergence if max_sims not reached

        # Option 2: Terminate if KL divergence between probability distributions stabilizes.
        # Requires self.kl_divergence to be calculated and valid (not inf).
        # Ensure enough simulations have run for KL divergence to be meaningful
        if self.mcts_sims > 50 and self.kl_divergence is not None and self.kl_divergence != math.inf:
             if self.kld_threshold >= self.kl_divergence:
                 # print(f"Terminating search due to KL divergence {self.kl_divergence} <= {self.kld_threshold} at sim {self.mcts_sims}")
                 return True

        # Continue search if neither condition met.
        return False
