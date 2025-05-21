import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import math # Added for math.inf and math.isnan

# Assuming mcts, checkersBoard, TDAgent are in the same directory or PYTHONPATH
import mcts
import checkersBoard
import TDAgent

# Mock for checkersBoard.CheckersBoard
class MockCheckersBoard:
    action_size = checkersBoard.CheckersBoard.action_size # Use actual action_size
    board_height = checkersBoard.CheckersBoard.board_height
    board_width = checkersBoard.CheckersBoard.board_width

    def __init__(self, initial_player=1):
        self.current_player = initial_player
        self._game_ended = False
        self._winner = 0
        self._valid_moves = [] # List of (MockCheckersBoard_move_instance, move_index)

    def get_valid_moves(self, player, include_index=True, include_chain_jumps=False):
        # Return pre-configured valid moves for the current player
        # Each move in self._valid_moves should be a tuple (board_after_move, move_index)
        # where board_after_move is another MockCheckersBoard instance.
        return [(move_board, idx) for move_board, idx in self._valid_moves if move_board.current_player != player] # Simplified logic, adjust as needed

    def game_ended(self):
        # Return pre-configured game end status and winner
        return self._game_ended, self._winner

    def set_game_ended(self, ended, winner):
        self._game_ended = ended
        self._winner = winner

    def set_valid_moves(self, player, moves_data):
        # moves_data: list of tuples (resulting_board_state_after_move, move_index)
        # The resulting_board_state_after_move should be a MockCheckersBoard instance
        # configured for the state after the move.
        self._valid_moves = []
        for board_after_move, move_idx in moves_data:
            self._valid_moves.append((board_after_move, move_idx))

    def __hash__(self):
        # Simple hash for testing purposes, might need adjustment if complex states are used
        return hash((self.current_player, self._game_ended, self._winner, tuple(m[1] for m in self._valid_moves)))

    def __eq__(self, other):
        if not isinstance(other, MockCheckersBoard):
            return False
        return (self.current_player == other.current_player and
                self._game_ended == other._game_ended and
                self._winner == other._winner and
                tuple(m[1] for m in self._valid_moves) == tuple(m[1] for m in other._valid_moves))


# Mock for the Neural Network
class MockNNet(torch.nn.Module): # Inherit from nn.Module
    def __init__(self):
        super().__init__() # Important for nn.Module
        # To satisfy: self.device = next(nnet_model.parameters()).device
        # We add a dummy parameter.
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        self.expected_value = 0.0
        self.expected_policy_logits = np.zeros(checkersBoard.CheckersBoard.action_size)

    def forward(self, features_tensor):
        # Return pre-configured value and policy logits
        # The MCTS class expects a tuple (value_tensor, policy_tensor)
        value_tensor = torch.tensor([self.expected_value], device=self.dummy_param.device)
        policy_tensor = torch.tensor(self.expected_policy_logits, device=self.dummy_param.device).unsqueeze(0)
        return value_tensor, policy_tensor

    def set_output(self, value, policy_logits):
        self.expected_value = value
        # Ensure policy_logits has the correct length
        if len(policy_logits) == checkersBoard.CheckersBoard.action_size:
            self.expected_policy_logits = np.array(policy_logits)
        else:
            raise ValueError(f"Policy logits length must be {checkersBoard.CheckersBoard.action_size}")

# Mock for TDAgent.extract_features
# This will be used with @patch typically, or we can make a callable mock.
def mock_extract_features(board, player):
    # Return a consistent, simple representation for a given board state.
    # The actual content doesn't matter as much as its consistency for mocking.
    # Using board hash or a predefined mapping if board states are complex.
    # For now, a simple fixed array.
    # The MCTS code expects shape (5, board_height, board_width)
    return np.zeros((5, checkersBoard.CheckersBoard.board_height, checkersBoard.CheckersBoard.board_width), dtype=np.float32)

# Basic Test Class
class TestMCTS(unittest.TestCase):
    def setUp(self):
        # Common setup for tests, e.g., creating mock objects
        self.mock_nnet = MockNNet()
        # self.mock_board = MockCheckersBoard() # Will be created per test or configured in setUp
        # self.mcts_instance = mcts.MCTS(self.mock_nnet) # Will be created per test

        # It's often better to patch TDAgent.extract_features globally for all tests in this class
        # or per test method if different mock behaviors are needed.
        self.patcher_extract_features = patch('TDAgent.extract_features', side_effect=mock_extract_features)
        self.mock_extract_features = self.patcher_extract_features.start()

    def tearDown(self):
        self.patcher_extract_features.stop()

    def test_mcts_initialization_defaults(self):
        # Test with default use_policy_head=True
        mcts_instance = mcts.MCTS(self.mock_nnet)
        self.assertEqual(mcts_instance.cpuct, 1.0)
        self.assertEqual(mcts_instance.e, 0.25)
        self.assertIs(mcts_instance.nnet_model, self.mock_nnet)
        self.assertTrue(mcts_instance.use_policy_head)
        self.assertEqual(mcts_instance.Qs, {})
        self.assertEqual(mcts_instance.Nsa, {})
        self.assertEqual(mcts_instance.Ns, {})
        self.assertEqual(mcts_instance.Es, {})
        self.assertEqual(mcts_instance.Ps, {})
        self.assertEqual(mcts_instance.mcts_sims, 0)
        self.assertIsNone(mcts_instance.max_sims)
        self.assertIsNone(mcts_instance.kld_threshold)
        self.assertIsNone(mcts_instance.kl_divergence) # Should be math.inf in get_probabilities, but None at init
        self.assertIsNone(mcts_instance.noise)
        self.assertEqual(mcts_instance.eval_batch_size, 5) # As per MCTS code
        self.assertEqual(mcts_instance.max_depth, 0)
        # Check device consistency
        self.assertEqual(mcts_instance.device, self.mock_nnet.dummy_param.device)

    def test_mcts_initialization_use_policy_head_false(self):
        mcts_instance = mcts.MCTS(self.mock_nnet, use_policy_head=False)
        self.assertFalse(mcts_instance.use_policy_head)
        # Other params should be default
        self.assertEqual(mcts_instance.cpuct, 1.0)
        self.assertIs(mcts_instance.nnet_model, self.mock_nnet)

    def test_mcts_initialization_custom_nnet(self):
        custom_nnet = MockNNet() # Create another instance
        mcts_instance = mcts.MCTS(custom_nnet)
        self.assertIs(mcts_instance.nnet_model, custom_nnet)

    def test_terminate_search_max_sims_reached(self):
        mcts_instance = mcts.MCTS(self.mock_nnet)
        mcts_instance.max_sims = 100
        mcts_instance.mcts_sims = 100
        self.assertTrue(mcts_instance.terminate_search())

        mcts_instance.mcts_sims = 101
        self.assertTrue(mcts_instance.terminate_search())

    def test_terminate_search_max_sims_not_reached(self):
        mcts_instance = mcts.MCTS(self.mock_nnet)
        mcts_instance.max_sims = 100
        mcts_instance.mcts_sims = 99
        # To test *only* max_sims not being reached as a reason to continue (assuming KLD also says continue):
        mcts_instance.kld_threshold = 0.01 
        mcts_instance.kl_divergence = 0.02 # KLD not met
        self.assertFalse(mcts_instance.terminate_search())


    def test_terminate_search_kld_met_sufficient_sims(self):
        mcts_instance = mcts.MCTS(self.mock_nnet)
        mcts_instance.kld_threshold = 0.01
        mcts_instance.kl_divergence = 0.005
        mcts_instance.mcts_sims = 51  # More than 50 sims
        self.assertTrue(mcts_instance.terminate_search())

    def test_terminate_search_kld_met_insufficient_sims(self):
        mcts_instance = mcts.MCTS(self.mock_nnet)
        mcts_instance.kld_threshold = 0.01
        mcts_instance.kl_divergence = 0.005
        mcts_instance.mcts_sims = 50  # Exactly 50 sims, so KLD check is skipped
        self.assertFalse(mcts_instance.terminate_search())

        mcts_instance.mcts_sims = 49 # Less than 50
        self.assertFalse(mcts_instance.terminate_search())

    def test_terminate_search_kld_not_met_sufficient_sims(self):
        mcts_instance = mcts.MCTS(self.mock_nnet)
        mcts_instance.kld_threshold = 0.01
        mcts_instance.kl_divergence = 0.015 # Higher than threshold
        mcts_instance.mcts_sims = 51
        self.assertFalse(mcts_instance.terminate_search())

    def test_terminate_search_no_max_sims_kld_met(self):
        mcts_instance = mcts.MCTS(self.mock_nnet)
        mcts_instance.max_sims = None
        mcts_instance.kld_threshold = 0.01
        mcts_instance.kl_divergence = 0.005
        mcts_instance.mcts_sims = 51
        self.assertTrue(mcts_instance.terminate_search())

    def test_terminate_search_no_max_sims_kld_not_met(self):
        mcts_instance = mcts.MCTS(self.mock_nnet)
        mcts_instance.max_sims = None
        mcts_instance.kld_threshold = 0.01
        mcts_instance.kl_divergence = 0.015
        mcts_instance.mcts_sims = 51
        self.assertFalse(mcts_instance.terminate_search())
        
    def test_terminate_search_max_sims_not_set_kld_not_set(self):
        mcts_instance = mcts.MCTS(self.mock_nnet)
        mcts_instance.max_sims = None
        mcts_instance.kld_threshold = 0.01 # kld_threshold is always set by get_probabilities
        mcts_instance.kl_divergence = None # Not calculated
        mcts_instance.mcts_sims = 100
        self.assertFalse(mcts_instance.terminate_search())

    def test_terminate_search_kld_is_inf(self):
        mcts_instance = mcts.MCTS(self.mock_nnet)
        mcts_instance.kld_threshold = 0.01
        mcts_instance.kl_divergence = float('inf') # KLD is infinity
        mcts_instance.mcts_sims = 51
        self.assertFalse(mcts_instance.terminate_search()) # Should not terminate if KLD is inf

    def test_terminate_search_max_sims_and_kld_met_max_sims_dominant(self):
        # If max_sims is met, it terminates regardless of KLD
        mcts_instance = mcts.MCTS(self.mock_nnet)
        mcts_instance.max_sims = 100
        mcts_instance.mcts_sims = 100
        mcts_instance.kld_threshold = 0.01
        mcts_instance.kl_divergence = 0.005 # KLD met
        # mcts_instance.mcts_sims = 100 # ensure this is the mcts_sims value at time of check # Duplicated
        self.assertTrue(mcts_instance.terminate_search())

    def test_terminate_search_max_sims_not_met_kld_met_kld_dominant(self):
        # If max_sims not met, KLD can cause termination
        mcts_instance = mcts.MCTS(self.mock_nnet)
        mcts_instance.max_sims = 100
        mcts_instance.mcts_sims = 51 # Sims count for KLD check
        mcts_instance.kld_threshold = 0.01
        mcts_instance.kl_divergence = 0.005 # KLD met
        self.assertTrue(mcts_instance.terminate_search())

    def test_search_expansion_new_node_use_policy_head_true(self):
        # Test expansion of a new node when using NN policy head
        initial_player = 1
        board = MockCheckersBoard(initial_player=initial_player)
        
        # Define valid moves for the initial board state
        dummy_next_board1 = MockCheckersBoard(initial_player=2) # Player changes after move
        dummy_next_board2 = MockCheckersBoard(initial_player=2)
        valid_moves_data = [
            (dummy_next_board1, 0), # (board_after_move, move_index)
            (dummy_next_board2, 1)
        ]
        board.set_valid_moves(initial_player, valid_moves_data)

        # Configure the mock neural network output for the initial board state
        expected_value = 0.5
        raw_policy_logits = np.random.rand(checkersBoard.CheckersBoard.action_size)
        self.mock_nnet.set_output(expected_value, raw_policy_logits)

        mcts_instance = mcts.MCTS(self.mock_nnet, use_policy_head=True)
        
        returned_value = mcts_instance.search(board, depth=0, dir_alpha=0)

        self.assertEqual(returned_value, expected_value, "Value from search should be NN's value on new node expansion.")

        state_key = self.mock_extract_features(board, initial_player).tobytes()
        self.assertIn(state_key, mcts_instance.Ps, "Priors (Ps) should be stored for the expanded state.")
        
        policy_tensor = torch.from_numpy(raw_policy_logits).float()
        # The MCTS code uses .cpu() before .numpy(), ensure our mock_nnet's tensors are on CPU or handle device if necessary.
        # MockNNet.forward already puts tensors on self.dummy_param.device.
        # MCTS.search does: policy_logits_tensor.cpu().numpy()
        # For this test, since mock_nnet.forward produces numpy arrays directly for its expected_policy_logits
        # and then wraps them in tensors, this should be fine.
        # Let's adjust to what MCTS does:
        # Ps[s] = self.nnet_model.process_policy(features_tensor, valids_mask).cpu().numpy()
        # MCTS.nnet_model.process_policy calls forward, then softmax, then masks, then normalizes.

        # Replicate policy processing from MCTS's NNetWrapper.process_policy
        # 1. Get raw policy from NNet
        # _, pi_logits_t = self.mock_nnet.forward(None) # features_tensor is not used by mock
        # pi_logits = pi_logits_t.cpu().numpy().flatten() # MCTS does flatten after cpu().numpy()
        # Our mock_nnet.set_output already takes numpy array and mock_nnet.forward returns it within a tensor.
        # The MCTS.search method gets pi from self.Ps[s] which is set during expansion.
        # The Ps[s] is set by:
        # features_tensor = torch.from_numpy(features).float().to(self.device)
        # value_tensor, policy_logits_tensor = self.nnet_model(features_tensor)
        # policy = torch.softmax(policy_logits_tensor, dim=1).squeeze()
        # valids_mask_tensor = torch.from_numpy(valids_mask).bool().to(self.device)
        # policy = policy * valids_mask_tensor
        # policy_sum = torch.sum(policy)
        # if policy_sum > 1e-8: # MCTS uses 1e-8
        #     policy /= policy_sum
        # else: # Uniform policy for valid moves if sum is too small
        #     # print("Warning: All valid moves have near zero probability. Using uniform distribution.")
        #     policy = torch.ones_like(policy) * valids_mask_tensor / torch.sum(valids_mask_tensor)
        # self.Ps[s] = policy.cpu().numpy()
        # v = value_tensor.item()

        # Simplified replication based on the logic:
        pi_from_nnet = torch.softmax(torch.from_numpy(raw_policy_logits).float(), dim=0).numpy()
        
        valids_mask = np.zeros(checkersBoard.CheckersBoard.action_size, dtype=bool) # MCTS uses bool then float for masking
        for _, idx in valid_moves_data:
            valids_mask[int(idx)] = True
        
        expected_pi = pi_from_nnet * valids_mask
        
        sum_expected_pi = np.sum(expected_pi)
        if sum_expected_pi > 1e-8: # MCTS uses 1e-8
            expected_pi /= sum_expected_pi
        else:
            if np.sum(valids_mask) > 0:
                 expected_pi = valids_mask.astype(np.float32) / np.sum(valids_mask) # Convert bool mask to float for division
            else: # No valid moves, policy should be all zero (or handle as per MCTS for this specific edge case)
                expected_pi = np.zeros_like(expected_pi, dtype=np.float32)


        np.testing.assert_array_almost_equal(mcts_instance.Ps[state_key], expected_pi,
                                             decimal=6, err_msg="Stored policy Ps is incorrect after expansion.")

        self.assertIn(state_key, mcts_instance.Ns, "State visit count Ns should be initialized for the expanded state.")
        self.assertEqual(mcts_instance.Ns[state_key], 0, "Ns for a newly expanded node should be 0 before any backup pass through it from a child.")
        
        self.assertNotIn(state_key, mcts_instance.Qs, "Qs should not be populated for actions of a newly expanded leaf.")
        self.assertNotIn(state_key, mcts_instance.Nsa, "Nsa should not be populated for actions of a newly expanded leaf.")

    def test_search_expansion_no_valid_moves_but_not_terminal(self):
        initial_player = 1
        board = MockCheckersBoard(initial_player=initial_player)
        board.set_valid_moves(initial_player, []) 
        board.set_game_ended(False, 0) 

        expected_value = -1.0 
        raw_policy_logits = np.random.rand(checkersBoard.CheckersBoard.action_size)
        self.mock_nnet.set_output(0.0, raw_policy_logits) 

        mcts_instance = mcts.MCTS(self.mock_nnet, use_policy_head=True)
        returned_value = mcts_instance.search(board, depth=0, dir_alpha=0)

        state_key = self.mock_extract_features(board, initial_player).tobytes()
        
        self.assertEqual(returned_value, expected_value, "Value should be -1 for no valid moves non-terminal state.")
        # In MCTS, if sum(valids_mask) == 0 and not ended:
        # self.Ps[s] is NOT set with NN output but rather directly to all zeros.
        # self.Es[s] is set to -1 (the returned value)
        # Ns[s] is set to 0
        self.assertIn(state_key, mcts_instance.Ps, "Ps should still be initialized even if no valid moves.")
        # The MCTS code in this scenario:
        # self.Ps[s] = np.zeros(self.action_size, dtype=np.float32)
        # self.Ns[s] = 0
        # self.Es[s] = -1 # loss for current player
        # return -1
        np.testing.assert_array_equal(mcts_instance.Ps[state_key], np.zeros(checkersBoard.CheckersBoard.action_size, dtype=np.float32),
                                      "Policy should be all zeros if no valid moves and not terminal.")
        self.assertEqual(mcts_instance.Ns.get(state_key, "Not Set"), 0, "Ns should be 0.")
        self.assertEqual(mcts_instance.Es.get(state_key, "Not Set"), -1.0, "Es should be -1.0.")


    def test_search_expansion_nn_returns_nan_inf_value(self):
        initial_player = 1
        board = MockCheckersBoard(initial_player=initial_player)
        valid_moves_data = [(MockCheckersBoard(initial_player=2), 0)]
        board.set_valid_moves(initial_player, valid_moves_data)

        # Test with NaN
        self.mock_nnet.set_output(float('nan'), np.array([1.0] + [0.0]*(checkersBoard.CheckersBoard.action_size -1)))
        mcts_instance_nan = mcts.MCTS(self.mock_nnet)
        returned_value_nan = mcts_instance_nan.search(board, depth=0, dir_alpha=0)
        # MCTS search: v = value_tensor.item(). If NN returns NaN, v becomes NaN.
        # This NaN value is then returned by the search function for a leaf node.
        self.assertTrue(math.isnan(returned_value_nan), "Value should be NaN if NN returns NaN and MCTS doesn't handle it before returning from leaf expansion.")

        # Test with Inf
        self.mock_nnet.set_output(float('inf'), np.array([1.0] + [0.0]*(checkersBoard.CheckersBoard.action_size -1)))
        mcts_instance_inf = mcts.MCTS(self.mock_nnet)
        returned_value_inf = mcts_instance_inf.search(board, depth=0, dir_alpha=0)
        # Similarly, Inf should propagate if not handled.
        self.assertEqual(returned_value_inf, float('inf'), "Value should be inf if NN returns inf and MCTS doesn't handle it before returning from leaf expansion.")

    def test_search_expansion_new_node_use_policy_head_false(self):
        # Test expansion of a new node when use_policy_head is False
        initial_player = 1
        board = MockCheckersBoard(initial_player=initial_player)
        
        # Define valid moves
        dummy_next_board1 = MockCheckersBoard(initial_player=2)
        dummy_next_board2 = MockCheckersBoard(initial_player=2)
        valid_moves_data = [
            (dummy_next_board1, 0), 
            (dummy_next_board2, 1)
        ]
        board.set_valid_moves(initial_player, valid_moves_data)

        # Configure mock NN (policy part won't be used for Ps, only value)
        expected_value = 0.3
        # This policy from NN will be ignored for Ps generation
        raw_policy_logits = np.random.rand(checkersBoard.CheckersBoard.action_size) 
        self.mock_nnet.set_output(expected_value, raw_policy_logits)

        mcts_instance = mcts.MCTS(self.mock_nnet, use_policy_head=False)
        
        returned_value = mcts_instance.search(board, depth=0, dir_alpha=0)

        # 1. Check returned value
        self.assertEqual(returned_value, expected_value, "Value from search should be NN's value even if policy head is not used.")

        # 2. Check Ps (priors)
        state_key = self.mock_extract_features(board, initial_player).tobytes()
        self.assertIn(state_key, mcts_instance.Ps, "Priors (Ps) should be stored for the expanded state.")
        
        # Expected policy should be uniform over valid moves
        expected_pi = np.zeros(checkersBoard.CheckersBoard.action_size)
        num_valid_moves = len(valid_moves_data)
        if num_valid_moves > 0:
            uniform_prob = 1.0 / num_valid_moves
            for _, idx in valid_moves_data:
                expected_pi[int(idx)] = uniform_prob
        
        np.testing.assert_array_almost_equal(mcts_instance.Ps[state_key], expected_pi,
                                             decimal=6, err_msg="Stored policy Ps should be uniform over valid moves when use_policy_head=False.")

        # 3. Check Ns
        self.assertIn(state_key, mcts_instance.Ns)
        self.assertEqual(mcts_instance.Ns[state_key], 0)

    def test_search_expansion_use_policy_head_false_no_valid_moves(self):
        # Test expansion with use_policy_head=False and no valid moves (but not terminal)
        initial_player = 1
        board = MockCheckersBoard(initial_player=initial_player)
        board.set_valid_moves(initial_player, []) # No valid moves
        board.set_game_ended(False, 0) # Not terminal

        expected_value = -1.0 # MCTS assigns -1 for this situation
        # NN output, value might be used if not for the no-valid-moves override
        self.mock_nnet.set_output(0.0, np.random.rand(checkersBoard.CheckersBoard.action_size))

        mcts_instance = mcts.MCTS(self.mock_nnet, use_policy_head=False)
        returned_value = mcts_instance.search(board, depth=0, dir_alpha=0)
        state_key = self.mock_extract_features(board, initial_player).tobytes()

        self.assertEqual(returned_value, expected_value, "Value should be -1 for no valid moves non-terminal state.")
        self.assertIn(state_key, mcts_instance.Ps, "Ps should be initialized.")
        # Policy should be all zeros as no moves are valid.
        expected_pi = np.zeros(checkersBoard.CheckersBoard.action_size)
        np.testing.assert_array_equal(mcts_instance.Ps[state_key], expected_pi,
                                      "Policy should be all zeros if no valid moves, even with use_policy_head=False.")
        self.assertEqual(mcts_instance.Ns.get(state_key, "Not Set"), 0)

    def _setup_board_and_mcts_for_backup_test(self, mcts_instance, initial_player=1):
        # Root board state
        self.board_s0 = MockCheckersBoard(initial_player=initial_player)
        self.s0_player = initial_player
        # To make keys distinct, we'll use different underlying board representations for extract_features
        # Let's assume mock_extract_features can be influenced by a property of the board if needed,
        # or that the player difference is enough. For now, we'll make the boards themselves "different"
        # in a way our mock_extract_features could distinguish if it were more complex.
        # Since mock_extract_features currently returns zeros, we will patch it per call in tests
        # if we need truly distinct features for s0 and s1 beyond player.
        # However, the MCTS keys are (features_bytes, player_id_in_key), so (zeros, 1) vs (zeros, 2) will be different.
        # The MCTS code uses `features.tobytes()` which does not include the player.
        # The state key `s` in MCTS is `features.tobytes()`. Player is handled implicitly by board state.
        # Let's ensure our mock_extract_features returns different features for different (board, player) pairs.
        # We will achieve this by having mock_extract_features return a unique array based on player.
        
        current_s0_features = np.zeros((5, checkersBoard.CheckersBoard.board_height, checkersBoard.CheckersBoard.board_width), dtype=np.float32)
        current_s0_features[0,0,0] = self.s0_player # Make features player-dependent for uniqueness
        self.s0_key = current_s0_features.tobytes()


        # Child board state after a move (e.g., move index 0)
        self.board_s1 = MockCheckersBoard(initial_player=3 - initial_player) # Opponent's turn
        self.s1_player = 3 - initial_player
        current_s1_features = np.zeros((5, checkersBoard.CheckersBoard.board_height, checkersBoard.CheckersBoard.board_width), dtype=np.float32)
        current_s1_features[0,0,0] = self.s1_player # Make features player-dependent
        self.s1_key = current_s1_features.tobytes()
        
        # s0 has one valid move leading to s1
        self.s0_valid_moves = [(self.board_s1, 0)] # (board_after_move, move_index)
        self.board_s0.set_valid_moves(self.s0_player, self.s0_valid_moves)
        
        # s1 has no further valid moves for this test (or is terminal) to simplify.
        self.board_s1.set_valid_moves(self.s1_player, []) 
        self.board_s1.set_game_ended(False, 0) # s1 is not terminal, will be expanded by NN

        # Configure NN output for s0 (root) - policy will select action 0
        s0_policy_logits = np.zeros(checkersBoard.CheckersBoard.action_size)
        s0_policy_logits[0] = 100 # Strongly prefer action 0
        
        # Mock extract_features to return specific features for S0 during its expansion
        self.mock_extract_features.side_effect = lambda board, player: current_s0_features if player == self.s0_player else current_s1_features

        self.mock_nnet.set_output(0.1, s0_policy_logits) # value_s0=0.1, policy_s0 for S0 expansion
        
        mcts_instance.search(self.board_s0, depth=0, dir_alpha=0) # Expands S0
        
        # Now, configure NN output for s1 (the node whose value will be backed up)
        self.s1_nn_value = 0.6  # Value of state s1 from s1's perspective
        s1_policy_logits = np.zeros(checkersBoard.CheckersBoard.action_size) 
        self.mock_nnet.set_output(self.s1_nn_value, s1_policy_logits) # For S1 expansion


    def test_search_backup_single_step(self):
        mcts_instance = mcts.MCTS(self.mock_nnet, use_policy_head=True)
        initial_player_s0 = 1
        self._setup_board_and_mcts_for_backup_test(mcts_instance, initial_player=initial_player_s0)
        
        # s0_key and s1_key are now attributes from the setup method.
        # The mock_extract_features is already configured by setup to return correct features for S0 and S1.
        # The mock_nnet is also set to return S1's value (self.s1_nn_value) when S1 is processed.

        # This search call will select action 0 from S0, then expand S1, then backup.
        mcts_instance.search(self.board_s0, depth=0, dir_alpha=0)

        expected_value_for_s0_action = -self.s1_nn_value # Value is from opponent's perspective

        self.assertIn(self.s0_key, mcts_instance.Qs)
        # The action in Qs is stored by the *action index*, not the state key of the next state.
        # MCTS code: self.Qs[s][a] = ... ; self.Nsa[s][a] = ...
        # The action 'a' is the move index, which is 0 in our case.
        action_idx = 0 
        self.assertIn(action_idx, mcts_instance.Qs[self.s0_key])
        self.assertAlmostEqual(mcts_instance.Qs[self.s0_key][action_idx], expected_value_for_s0_action, delta=1e-6)

        self.assertIn(self.s0_key, mcts_instance.Nsa)
        self.assertIn(action_idx, mcts_instance.Nsa[self.s0_key])
        self.assertEqual(mcts_instance.Nsa[self.s0_key][action_idx], 1)

        self.assertEqual(mcts_instance.Ns[self.s0_key], 1) # Incremented during backup
                         
        self.assertIn(self.s1_key, mcts_instance.Ns)
        self.assertEqual(mcts_instance.Ns[self.s1_key], 0) # S1 is expanded, Ns is 0.


    def test_search_backup_two_steps(self):
        mcts_instance = mcts.MCTS(self.mock_nnet, use_policy_head=True)
        initial_player_s0 = 1
        self._setup_board_and_mcts_for_backup_test(mcts_instance, initial_player=initial_player_s0)
        action_idx = 0 # The action leading from s0 to s1

        # First pass (already done by setup for S0 expansion, now the first *selection* pass)
        self.mock_extract_features.side_effect = lambda board, player: \
            current_s0_features if player == self.s0_player else current_s1_features \
            if hasattr(self, 's0_player') and player == getattr(self, 's0_player', None) else \
            current_s1_features if hasattr(self, 's1_player') and player == getattr(self, 's1_player', None) else \
            np.random.rand(5, checkersBoard.CheckersBoard.board_height, checkersBoard.CheckersBoard.board_width) # default for safety

        # Features for S0 and S1 need to be consistent for keys.
        current_s0_features = np.zeros((5, checkersBoard.CheckersBoard.board_height, checkersBoard.CheckersBoard.board_width), dtype=np.float32)
        current_s0_features[0,0,0] = initial_player_s0
        current_s1_features = np.zeros((5, checkersBoard.CheckersBoard.board_height, checkersBoard.CheckersBoard.board_width), dtype=np.float32)
        current_s1_features[0,0,0] = 3 - initial_player_s0
        
        self.mock_extract_features.side_effect = lambda board, player: \
            current_s0_features if board == self.board_s0 else current_s1_features

        # NN for S1 expansion
        s1_policy_logits = np.zeros(checkersBoard.CheckersBoard.action_size)
        self.mock_nnet.set_output(self.s1_nn_value, s1_policy_logits)
        mcts_instance.search(self.board_s0, depth=0, dir_alpha=0)
        
        q_after_first_pass = -self.s1_nn_value
        self.assertEqual(mcts_instance.Nsa[self.s0_key][action_idx], 1)
        self.assertEqual(mcts_instance.Ns[self.s0_key], 1)
        self.assertAlmostEqual(mcts_instance.Qs[self.s0_key][action_idx], q_after_first_pass, delta=1e-6)

        # NN for S1 expansion (still the same for the second pass)
        self.mock_nnet.set_output(self.s1_nn_value, s1_policy_logits)
        # Second pass
        mcts_instance.search(self.board_s0, depth=0, dir_alpha=0)

        expected_q_after_second_pass = (q_after_first_pass + (-self.s1_nn_value)) / 2.0
        
        self.assertAlmostEqual(mcts_instance.Qs[self.s0_key][action_idx], expected_q_after_second_pass, delta=1e-6)
        self.assertEqual(mcts_instance.Nsa[self.s0_key][action_idx], 2)
        self.assertEqual(mcts_instance.Ns[self.s0_key], 2)
        self.assertEqual(mcts_instance.Ns[self.s1_key], 0) # S1 is always a new expansion

    def _setup_board_and_mcts_for_backup_test(self, mcts_instance, initial_player=1):
        # Root board state
        self.board_s0 = MockCheckersBoard(initial_player=initial_player)
        self.s0_player = initial_player
        
        # Child board state after a move (e.g., move index 0)
        self.board_s1 = MockCheckersBoard(initial_player=3 - initial_player) # Opponent's turn
        self.s1_player = 3 - initial_player

        # Make features distinct for s0 and s1 using a patch for mock_extract_features
        # This ensures s0_key and s1_key are different.
        s0_features_arr = np.array([initial_player], dtype=np.float32).reshape(1,1,1) # Minimal unique feature
        s1_features_arr = np.array([3-initial_player], dtype=np.float32).reshape(1,1,1) # Minimal unique feature
        
        # Pad to match expected shape (5, H, W) if necessary, or adjust mock_extract_features globally for this.
        # For simplicity, we'll assume extract_features can return variable shapes, and tobytes() handles it.
        # Or, more robustly, ensure the features have the correct shape.
        # Let's use the existing global mock_extract_features and rely on the (board,player) tuple to be unique for its (future) more complex version.
        # For now, we will explicitly create different feature arrays.
        
        mock_s0_features = np.zeros((5, checkersBoard.CheckersBoard.board_height, checkersBoard.CheckersBoard.board_width), dtype=np.float32)
        mock_s0_features[0, 0, 0] = initial_player # Differentiate features based on player

        mock_s1_features = np.zeros((5, checkersBoard.CheckersBoard.board_height, checkersBoard.CheckersBoard.board_width), dtype=np.float32)
        mock_s1_features[0, 0, 0] = 3 - initial_player # Differentiate features

        self.s0_key = mock_s0_features.tobytes()
        self.s1_key = mock_s1_features.tobytes()
        
        # Configure self.mock_extract_features (the patched version) to return these specific features
        def side_effect_extract_features(board, player):
            if board == self.board_s0 and player == self.s0_player:
                return mock_s0_features
            elif board == self.board_s1 and player == self.s1_player:
                return mock_s1_features
            # Fallback for any other unexpected calls - though tests should be specific
            return np.random.rand(5, checkersBoard.CheckersBoard.board_height, checkersBoard.CheckersBoard.board_width).astype(np.float32)

        self.mock_extract_features.side_effect = side_effect_extract_features
        
        # s0 has one valid move leading to s1
        self.action_s0_to_s1 = 0 # This is the action index
        self.s0_valid_moves = [(self.board_s1, self.action_s0_to_s1)] 
        self.board_s0.set_valid_moves(self.s0_player, self.s0_valid_moves)
        
        self.board_s1.set_valid_moves(self.s1_player, []) 
        self.board_s1.set_game_ended(False, 0) 

        s0_policy_logits = np.zeros(checkersBoard.CheckersBoard.action_size)
        s0_policy_logits[self.action_s0_to_s1] = 100 
        self.mock_nnet.set_output(0.1, s0_policy_logits) 
        
        mcts_instance.search(self.board_s0, depth=0, dir_alpha=0) # Expands S0
        
        self.s1_nn_value = 0.6  
        s1_policy_logits = np.zeros(checkersBoard.CheckersBoard.action_size)
        self.mock_nnet.set_output(self.s1_nn_value, s1_policy_logits)


    def test_search_backup_single_step(self):
        mcts_instance = mcts.MCTS(self.mock_nnet, use_policy_head=True)
        initial_player_s0 = 1
        self._setup_board_and_mcts_for_backup_test(mcts_instance, initial_player=initial_player_s0)
        
        mcts_instance.search(self.board_s0, depth=0, dir_alpha=0)

        expected_value_for_s0_action = -self.s1_nn_value

        self.assertIn(self.s0_key, mcts_instance.Qs)
        self.assertIn(self.action_s0_to_s1, mcts_instance.Qs[self.s0_key])
        self.assertAlmostEqual(mcts_instance.Qs[self.s0_key][self.action_s0_to_s1], expected_value_for_s0_action,
                               delta=1e-6, msg="Q-value for (s0, action_to_s1) is incorrect after backup.")

        self.assertIn(self.s0_key, mcts_instance.Nsa)
        self.assertIn(self.action_s0_to_s1, mcts_instance.Nsa[self.s0_key])
        self.assertEqual(mcts_instance.Nsa[self.s0_key][self.action_s0_to_s1], 1)

        self.assertEqual(mcts_instance.Ns[self.s0_key], 1)
                         
        self.assertIn(self.s1_key, mcts_instance.Ns)
        self.assertEqual(mcts_instance.Ns[self.s1_key], 0)


    def test_search_backup_two_steps(self):
        mcts_instance = mcts.MCTS(self.mock_nnet, use_policy_head=True)
        initial_player_s0 = 1
        self._setup_board_and_mcts_for_backup_test(mcts_instance, initial_player=initial_player_s0)

        # First pass (triggers backup)
        mcts_instance.search(self.board_s0, depth=0, dir_alpha=0)
        
        q_after_first_pass = -self.s1_nn_value
        self.assertEqual(mcts_instance.Nsa[self.s0_key][self.action_s0_to_s1], 1)
        self.assertEqual(mcts_instance.Ns[self.s0_key], 1)
        self.assertAlmostEqual(mcts_instance.Qs[self.s0_key][self.action_s0_to_s1], q_after_first_pass, delta=1e-6)

        # Ensure NN is set for S1's (re)-expansion in the second pass
        s1_policy_logits = np.zeros(checkersBoard.CheckersBoard.action_size)
        self.mock_nnet.set_output(self.s1_nn_value, s1_policy_logits) 

        # Second pass
        mcts_instance.search(self.board_s0, depth=0, dir_alpha=0)

        # Q value is averaged: (Nsa_old * Q_old + new_value) / Nsa_new
        # Nsa_old for this action was 1. Q_old was q_after_first_pass.
        # new_value is -self.s1_nn_value. Nsa_new is 2.
        expected_q_after_second_pass = (1 * q_after_first_pass + (-self.s1_nn_value)) / 2.0
        
        self.assertAlmostEqual(mcts_instance.Qs[self.s0_key][self.action_s0_to_s1], expected_q_after_second_pass,
                               delta=1e-6, msg="Q-value for (s0, action_to_s1) is incorrect after second backup.")
        self.assertEqual(mcts_instance.Nsa[self.s0_key][self.action_s0_to_s1], 2)
        self.assertEqual(mcts_instance.Ns[self.s0_key], 2)
        self.assertEqual(mcts_instance.Ns[self.s1_key], 0) 

if __name__ == '__main__':
    unittest.main()
