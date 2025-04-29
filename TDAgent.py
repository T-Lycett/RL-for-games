import checkersBoard
import numpy as np
import math
import random
from random import shuffle
import torch
import resNN
import multiprocessing
import threading
import time
import mcts
from utils import weighted_pick
from multiprocessing import Lock, Value
from functools import partial

board_height = checkersBoard.CheckersBoard.board_height
board_width = checkersBoard.CheckersBoard.board_width


def evaluate(model, board, current_player):
    features = extract_features(board, current_player)
    features = torch.from_numpy(features).float().unsqueeze(0)
    device = next(model.parameters()).device
    features = features.to(device)
    with torch.no_grad():
        return model(features)[0].cpu().numpy()


def flip_pieces(pieces):
    pieces = pieces[::-1, :]
    pieces = pieces[:, ::-1]
    return pieces


def extract_features(board, current_player):
    players_pieces = np.copy(board.get_player_positions(current_player))
    opp_pieces = np.copy(board.get_player_positions(-current_player))
    players_kings = np.copy(board.get_players_kings(current_player))
    opp_kings = np.copy(board.get_players_kings(-current_player))
    if current_player == -1:
        players_pieces = flip_pieces(players_pieces)
        opp_pieces = flip_pieces(opp_pieces)
        players_kings = flip_pieces(players_kings)
        opp_kings = flip_pieces(opp_kings)
    moves_until_draw = np.zeros((board_height, board_width)) + (50 - board.moves_without_capture) / 50
    # Stack features along axis 0 for channels_first format (C, H, W)
    return np.stack([players_pieces, opp_pieces, players_kings, opp_kings, moves_until_draw], axis=0)


def get_move(board, player, mcts_instance, kld_threshold, temperature, max_sims=None):
    moves = board.get_valid_moves(player, include_index=True, include_chain_jumps=False)
    num_moves = len(moves)
    if num_moves == 0:
        return None, None
    if num_moves == 1:
        mcts_instance.mcts_sims = 0
        exponentiated_probs = np.zeros((checkersBoard.CheckersBoard.action_size))
        index = int(moves[0][1])
        exponentiated_probs[index] = 1
        if sum(exponentiated_probs) > 1.01:
            print(exponentiated_probs)
        return moves[0][0], exponentiated_probs
    else:
        exponentiated_probs, node_probs = mcts_instance.get_probabilities(board, player, kld_threshold=kld_threshold, max_sims=max_sims, temperature=temperature, dir_alpha=1.75)
        choices = np.ndarray((checkersBoard.CheckersBoard.action_size), dtype=checkersBoard.CheckersBoard)
        for m, i in moves:
            choices[int(i)] = m
        move_index = np.random.multinomial(1, exponentiated_probs)
        move_index = np.where(move_index == 1)[0][0]
        chosen_move = choices[move_index]
        return chosen_move, node_probs


def self_play_init(l, val, width, res_blocks, q_learning_only):
    global self_play_lock
    self_play_lock = l
    global games_to_play
    games_to_play = val
    global model_width
    model_width = width
    global model_res_blocks
    model_res_blocks = res_blocks
    global model_q_learning_only
    model_q_learning_only = q_learning_only


def self_play_game_player(model_filename, kld_threshold, q_learning):
    try:
        # Create and load a PyTorch model
        device = torch.device("cpu")  # Worker processes usually use CPU
        
        # Get model parameters from global variables (passed during initialization)
        width = model_width if 'model_width' in globals() else 64  # Default width=64
        residual_blocks = model_res_blocks if 'model_res_blocks' in globals() else 3  # Default blocks=3
        q_learning_only = model_q_learning_only if 'model_q_learning_only' in globals() else q_learning  # Default based on input param
        
        # Initialize model with the right parameters
        model = resNN.ResNN(width=width, residual_blocks=residual_blocks, q_learning_only=q_learning_only)
        
        try:
            model.load_state_dict(torch.load(model_filename, map_location=device))
            model.to(device)
            model.eval()  # Set to evaluation mode
        except Exception as e:
            print(f"ERROR loading model '{model_filename}': {e}")
            import traceback
            traceback.print_exc()
            return []  # Return empty training examples on model loading error
            
        max_moves_until_t0 = 30
        training_examples = []
        games = 0
        game_id = -1
        
        while True:
            try:
                with self_play_lock:
                    print('games left: ' + str(games_to_play.value))
                    if games_to_play.value == 0:
                        break
                    else:
                        games_to_play.value -= 1
                        game_id = games_to_play.value
                        print('started game ' + str(game_id))
                        
                mcts_instance = mcts.MCTS(model, use_policy_head=not q_learning)
                games += 1
                move_history = []
                game_ended = False
                board = checkersBoard.CheckersBoard(start_positions=True)
                current_player = 1
                # state = extract_features(board, current_player)
                # move_history.append([state, False, None, None])
                num_moves = 0
                moves_until_t0 = math.inf # random.randint(1, max_moves_until_t0)
                num_mcts_sims = []
                max_moves = 200  # Safety limit to prevent infinite games
                
                while not game_ended and num_moves < max_moves:
                    try:
                        if num_moves < moves_until_t0:
                            move, probs = get_move(board, current_player, mcts_instance, kld_threshold=kld_threshold, temperature=0.2)
                        else:
                            move, probs = get_move(board, current_player, mcts_instance, kld_threshold, temperature=0.05)
                            
                        if move is None or np.all(np.isnan(probs)):
                            print(f"ERROR in game {game_id}: Invalid move or probabilities. Ending game.")
                            game_ended = True
                            break
                            
                        state = extract_features(board, current_player)
                        
                        # Validate returned probabilities
                        if np.any(np.isnan(probs)):
                            print(f"WARNING in game {game_id}: NaN values in probabilities")
                            probs = np.nan_to_num(probs, nan=0.0)
                            if np.sum(probs) < 1e-10:
                                # If sum is too small after removing NaNs, use uniform
                                num_valid = np.count_nonzero(probs > 0)
                                if num_valid > 0:
                                    probs[probs > 0] = 1.0 / num_valid
                                else:
                                    probs = np.ones_like(probs) / len(probs)
                                    
                        # Ensure probabilities sum to 1 (approximately)
                        if abs(np.sum(probs) - 1.0) > 0.01:
                            if np.sum(probs) > 0:
                                probs = probs / np.sum(probs)
                            else:
                                probs = np.ones_like(probs) / len(probs)
                        
                        move_history.append([state, current_player, None, probs])
                        
                        if mcts_instance.mcts_sims != 0:
                            num_mcts_sims.append(mcts_instance.mcts_sims)
                            
                        # Apply the move
                        board.set_positions(move)
                        game_ended, winner = board.game_ended()
                        current_player = board.current_player
                        num_moves += 1
                        
                    except Exception as e:
                        print(f"ERROR in game {game_id}, move {num_moves}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Try to recover and continue or end the game
                        game_ended = True
                        winner = 0  # Draw on error
                
                # Game ended - handle the result    
                if num_moves >= max_moves:
                    print(f"WARNING: Game {game_id} reached move limit. Ending as draw.")
                    winner = 0
                
                # Finalize game history
                if winner == 1:
                    winner_str = 'player 1'
                elif winner == -1:
                    winner_str = 'player 2'
                else:
                    winner_str = 'draw'
                    
                avg_mcts_sims = sum(num_mcts_sims) / len(num_mcts_sims) if num_mcts_sims else 0
                max_mcts_sims = max(num_mcts_sims) if num_mcts_sims else 0
                min_mcts_sims = min(num_mcts_sims) if num_mcts_sims else 0
                
                print(f'finished game {game_id}, temp = 0 at move {moves_until_t0}, game length: {num_moves}, ' +
                      f'average mcts sims: {avg_mcts_sims}, max mcts sims: {max_mcts_sims}, ' +
                      f'min mcts sims: {min_mcts_sims}, winner: {winner_str}')
                
                # Record final state if needed
                if not move_history:
                    # Game ended with no moves - skip
                    continue
                    
                # Add final result to all states in history
                try:
                    for i, (_, cur_player, r, _) in enumerate(move_history):
                        move_history[i][2] = cur_player * winner
                        
                    for m in move_history:
                        training_examples.append(m)
                except Exception as e:
                    print(f"ERROR finalizing game {game_id} history: {e}")
                    import traceback
                    traceback.print_exc()
                    
            except Exception as e:
                print(f"CRITICAL ERROR in game {game_id}: {e}")
                import traceback
                traceback.print_exc()
                continue  # Try to process next game
                
        # Clean up
        try:
            del model
        except Exception as e:
            print(f"ERROR cleaning up resources: {e}")
        
        return training_examples
        
    except Exception as e:
        print(f"FATAL ERROR in self_play_game_player: {e}")
        import traceback
        traceback.print_exc()
        return []  # Return empty list on fatal error


class TDAgent():
    def __init__(self, model_filename, learner=True, lr=0.0001, search_depth=3, q_learning=True):
        self.q_learning = q_learning
        self.learner = learner
        self.lr = lr
        self.search_depth = search_depth
        self.training_examples = []
        self.model_filename = model_filename
        
        # Model architecture parameters
        self.width = 64  # Default width
        self.residual_blocks = 3  # Default blocks
        
        # Set up device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize the PyTorch model
        self.NN = resNN.ResNN(width=self.width, residual_blocks=self.residual_blocks, q_learning_only=q_learning)
        
        try:
            self.NN.load_state_dict(torch.load(model_filename, map_location=self.device))
            print(f"Successfully loaded model from {model_filename}")
        except Exception as e:
            print(f"Error loading model: {e}. Will initialize a new model.")
            # If loading fails, we'll just keep the newly initialized model
            
        self.NN.to(self.device)
        self.NN.eval()  # Start in evaluation mode
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.NN.parameters(), lr=lr)
            
        if self.learner:
            self.game_players = 4  # int(multiprocessing.cpu_count() / 2)

    def set_lr(self, lr):
        self.lr = lr
        # Update the optimizer with the new learning rate
        self.optimizer = torch.optim.Adam(self.NN.parameters(), lr=lr)
        print('set learning rate to ' + str(lr))
        self.save_model(self.model_filename)

    @staticmethod
    def extract_features(board, current_player):
        players_pieces = np.copy(board.get_player_positions(current_player))
        opp_pieces = np.copy(board.get_player_positions(-current_player))
        players_kings = np.copy(board.get_players_kings(current_player))
        opp_kings = np.copy(board.get_players_kings(-current_player))
        if current_player == -1:
            players_pieces = flip_pieces(players_pieces)
            opp_pieces = flip_pieces(opp_pieces)
            players_kings = flip_pieces(players_kings)
            opp_kings = flip_pieces(opp_kings)
        moves_until_draw = np.zeros((board_height, board_width)) + (50 - board.moves_without_capture) / 50
        # Stack features along axis 0 for channels_first format (C, H, W)
        return np.stack([players_pieces, opp_pieces, players_kings, opp_kings, moves_until_draw], axis=0)

    @staticmethod
    def flip_pieces(pieces):
        pieces = pieces[::-1, :]
        pieces = pieces[:, ::-1]
        return pieces

    def get_move(self, board, player, mcts_instance, kld_threshold):
        moves = board.get_valid_moves(player, include_index=True, include_chain_jumps=False)
        num_moves = len(moves)
        if num_moves == 0:
            return None, None
        if num_moves == 1:
            mcts_instance.mcts_sims = 0
            nn_eval = self.evaluate(moves[0][0], moves[0][0].current_player)[0]
            if player != moves[0][0].current_player:
                nn_eval *= -1
            return moves[0][0], nn_eval
        else:
            exponentiated_probs, _ = mcts_instance.get_probabilities(board, player, kld_threshold=kld_threshold, temperature=0, verbose=False, dir_alpha=0)
            choices = np.ndarray(checkersBoard.CheckersBoard.action_size, dtype=checkersBoard.CheckersBoard)
            for move, i in moves:
                choices[int(i)] = move
                
            # Ensure probabilities are valid for weighted picking
            if np.sum(exponentiated_probs) < 1e-10 or np.isnan(exponentiated_probs).any():
                print("Warning: Invalid probability distribution. Using uniform selection.")
                valid_indices = [int(i) for _, i in moves]
                selected_index = random.choice(valid_indices)
                move = choices[selected_index]
            else:
                try:
                    move = choices[weighted_pick(exponentiated_probs)]
                except Exception as e:
                    print(f"Error during move selection: {e}")
                    # Fallback to first move
                    move = moves[0][0]

            current_state = self.extract_features(board, board.current_player).tobytes()
            move_state = self.extract_features(move, move.current_player).tobytes()
            nn_val = self.evaluate(move, move.current_player)[0]
            
            # Safely access MCTS Q-values
            mcts_val = mcts_instance.Qs.get(current_state, "Unknown")
            
            if player != move.current_player:
                nn_val *= -1
            eval_str = 'Neural Network: ' + str(nn_val) + ', MCTS: ' + str(mcts_val)
            return move, eval_str

    def update_model(self, moves, lambda_val, batch_size):
        print('updating neural network...')
        losses = []
        for i, move in enumerate(moves):
            state, _, reward, probs = move
            losses.append([state, reward, probs])
        states, targets, probs = list(zip(*losses))
        assert not np.all(np.isnan(probs))
        
        # Convert data to PyTorch tensors
        states = np.asarray(states)
        targets = np.asarray(targets)
        
        # Ensure proper shape
        if states.shape[1:] != (5, board_height, board_width):  # channels_first format
            print(f"Warning: Unexpected state shape: {states.shape}. Expected ({batch_size}, 5, {board_height}, {board_width})")
            try:
                # Reshape to (batch_size, channels, height, width)
                states = states.reshape((batch_size, 5, board_height, board_width))
                print(f"Reshaped states to: {states.shape}")
            except ValueError as e:
                print(f"Error reshaping states: {e}. Check feature extraction and batch assembly.")
                raise e
                
        # Convert to PyTorch tensors and move to device
        states_tensor = torch.from_numpy(states).float().to(self.device)
        targets_tensor = torch.from_numpy(targets).float().to(self.device)
        
        # Set model to training mode
        self.NN.train()
        
        # Training loop
        self.optimizer.zero_grad()
        
        if self.q_learning:
            # Only predict and train on value
            pred_value = self.NN(states_tensor)
            if isinstance(pred_value, tuple):
                pred_value = pred_value[0]  # Extract value if model returns (value, policy)
            
            # Define loss function
            value_loss_fn = torch.nn.MSELoss()
            loss = value_loss_fn(pred_value.squeeze(), targets_tensor)
        else:
            # Predict both value and policy
            probs = np.reshape(probs, (batch_size, checkersBoard.CheckersBoard.action_size))
            probs_tensor = torch.from_numpy(probs).float().to(self.device)
            
            pred_value, pred_probs = self.NN(states_tensor)
            
            # Define loss functions
            value_loss_fn = torch.nn.MSELoss()
            policy_loss_fn = torch.nn.CrossEntropyLoss() if pred_probs.shape[1:] == probs_tensor.shape[1:] else torch.nn.KLDivLoss(reduction='batchmean')
            
            # Calculate losses
            value_loss = value_loss_fn(pred_value.squeeze(), targets_tensor)
            
            # For policy loss, check if we need log_softmax
            if isinstance(policy_loss_fn, torch.nn.KLDivLoss):
                pred_probs = torch.nn.functional.log_softmax(pred_probs, dim=1)
                policy_loss = policy_loss_fn(pred_probs, probs_tensor)
            else:
                policy_loss = policy_loss_fn(pred_probs, probs_tensor)
                
            loss = value_loss + policy_loss
        
        # Backpropagation and optimization
        loss.backward()
        self.optimizer.step()
        
        # Set model back to evaluation mode
        self.NN.eval()
        
        print(f"Training loss: {loss.item()}")

    def self_play(self, kld_threshold, num_games=1000, iterations=1, lambda_val=0.9, batch_size=512):
        if not self.learner:
            return False
            
        for iteration in range(iterations):
            print('iteration: ' + str(iteration))
            games_left_to_play = Value('i', num_games)
            
            # Clean up existing model before multiprocessing
            try:
                # PyTorch doesn't need explicit session cleanup like TensorFlow
                self.NN.cpu()  # Move model to CPU before multiprocessing
                model_copy = self.NN.state_dict()  # Make a copy of the state dict
                self.save_model(self.model_filename)  # Save the model for workers to load
                print("Successfully prepared model for worker processes")
            except Exception as e:
                print(f"Warning during model preparation: {e}")
                
            lock = Lock()
            game_player_pool = None
            results = []
            
            try:
                # Create the process pool with model parameters
                game_player_pool = multiprocessing.Pool(
                    processes=self.game_players, 
                    initializer=self_play_init, 
                    initargs=(lock, games_left_to_play, self.width, self.residual_blocks, self.q_learning)
                )
                
                # Apply the game player function asynchronously
                results = [
                    game_player_pool.apply_async(
                        self_play_game_player, 
                        args=(self.model_filename, kld_threshold, self.q_learning)
                    ) for _ in range(self.game_players)
                ]
                
                # Set a timeout for getting results (30 minutes per worker)
                timeout_per_worker = 1800  
                game_player_pool.close()
                
                # Safely collect results with timeout
                self_play_results = []
                for i, r in enumerate(results):
                    try:
                        result = r.get(timeout=timeout_per_worker)
                        self_play_results.append(result)
                    except multiprocessing.TimeoutError:
                        print(f"Worker {i} timed out after {timeout_per_worker} seconds")
                    except Exception as e:
                        print(f"Error getting results from worker {i}: {e}")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                print(f"Critical error in self_play multiprocessing: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Ensure pool is properly cleaned up
                if game_player_pool:
                    try:
                        game_player_pool.terminate()
                        game_player_pool.join()
                    except:
                        print("Error during pool cleanup")
            
            # Process the results from successful workers
            for sublist in self_play_results:
                if not sublist:
                    print("Warning: Empty result from worker")
                    continue
                    
                for example in sublist:
                    self.training_examples.append(example)
            
            print('training examples: ' + str(len(self.training_examples)))
            
            # Train the model if we have enough examples
            if len(self.training_examples) > batch_size:
                try:
                    # Move model back to the training device
                    self.NN.to(self.device)
                    
                    # Deduplicate and shuffle training data
                    self.training_examples = self.deduplicate_training_data(self.training_examples)
                    shuffle(self.training_examples)
                    
                    # Train in batches
                    training_batches = 0
                    while len(self.training_examples) > batch_size:
                        try:
                            batch = self.training_examples[-batch_size:]
                            self.update_model(batch, lambda_val, batch_size)
                            self.training_examples[-batch_size:] = []
                            training_batches += 1
                            
                            # Save after each few batches as checkpoint
                            if training_batches % 5 == 0:
                                self.save_model(self.model_filename)
                                
                        except Exception as e:
                            print(f"Error during model update: {e}")
                            import traceback
                            traceback.print_exc()
                            break  # Exit training loop on error
                    
                    # Final save
                    self.save_model(self.model_filename)
                    
                except Exception as e:
                    print(f"Critical error during training: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Ensure model is in the right state after training
        self.NN.to(self.device)
        self.NN.eval()

    def calculate_kld_threshold(self, current_threshold, average_game_length):
        target_game_length = 125

    @staticmethod
    def position_evaluator(evaluation_queue, evaluated_positions, model_filename):
        device = torch.device("cpu")  # Worker process typically uses CPU
        model = resNN.ResNN()  # Initialize with proper parameters
        model.load_state_dict(torch.load(model_filename, map_location=device))
        model.to(device)
        model.eval()
        
        while True:
            board, player, p_id = evaluation_queue.get()
            TDAgent.evaluate_position(model, board, player, evaluated_positions[p_id])

    @staticmethod
    def evaluate_position(model, board, current_player, evaluated_positions):
        features = TDAgent.extract_features(board, current_player)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
        device = next(model.parameters()).device
        features_tensor = features_tensor.to(device)
        
        with torch.no_grad():
            prediction = model(features_tensor)
            # Handle potential tuple return (value, policy)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            # Convert back to numpy for storage
            evaluated_positions[features.tobytes()] = prediction.cpu().numpy()

    @staticmethod
    def deduplicate_training_data(training_data):
        print('number of positions before deduplicate: ' + str(len(training_data)))
        new_data = {}
        counts = {}
        for state, player, reward, probs in training_data:
            hashable_state = state.tobytes()
            if hashable_state not in new_data:
                new_data[hashable_state] = [state, player, reward, probs]
                counts[hashable_state] = 1
            else:
                new_data[hashable_state][2] = (counts[hashable_state] * new_data[hashable_state][2] + reward) / (counts[hashable_state] + 1)
                counts[hashable_state] += 1
        new_training_examples = []
        for state, player, reward, probs in new_data.values():
            new_training_examples.append([state, player, reward, probs])
        print('number of positions after deduplicate: ' + str(len(new_training_examples)))
        return new_training_examples

    def evaluate(self, board, current_player):
        features = extract_features(board, current_player)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        self.NN.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():
            prediction = self.NN(features_tensor)
            # Handle potential tuple return (value, policy)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            return prediction.cpu().numpy()

    def save_model(self, filepath):
        torch.save(self.NN.state_dict(), filepath)
        print('neural network model saved to ' + self.model_filename)

    def load_model(self, filepath):
        self.model_filename = filepath
        self.NN.load_state_dict(torch.load(filepath, map_location=self.device))
        self.NN.to(self.device)
        self.NN.eval()
