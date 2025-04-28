import checkersBoard
import numpy as np
import math
import random
from random import shuffle
import cnn
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
    features = np.asarray([features])
    return model.predict(features)


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


def self_play_init(l, val):
    global self_play_lock
    self_play_lock = l
    global games_to_play
    games_to_play = val


def self_play_game_player(model_filename, kld_threshold, q_learning):
    try:
        # Set up TensorFlow session with error handling
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        
        # Handle potential TensorFlow session creation errors
        try:
            sess = tf.Session(config=config)
            keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
        except Exception as e:
            print(f"ERROR creating TensorFlow session: {e}")
            import traceback
            traceback.print_exc()
            return []  # Return empty training examples on session error
            
        # Handle potential model loading errors
        try:
            model = keras.models.load_model(model_filename)
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
            keras.backend.clear_session()
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
        # self.NN = resNN.ResNN()
        self.model_filename = model_filename
        # self.NN = cnn.CNN()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        self.sess = tf.Session(config=config)
        keras.backend.set_session(self.sess)  # set this TensorFlow session as the default session for Keras
        self.NN = keras.models.load_model(model_filename)
        # self.NN.compile(keras.optimizers.Adam(lr=lr), loss=tf.losses.mean_squared_error)
        if self.learner:
            self.game_players = 4  # int(multiprocessing.cpu_count() / 2)

    def set_lr(self, lr):
        self.lr = lr
        if self.q_learning:
            self.NN.compile(keras.optimizers.Adam(lr=lr), loss=keras.losses.mean_squared_error)
        else:
            self.NN.compile(keras.optimizers.Adam(lr=lr), loss=[keras.losses.mean_squared_error, keras.losses.categorical_crossentropy])
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
        states = np.asarray(states)
        targets = np.asarray(targets)
        if states.shape[1:] != (board_height, board_width, 5):
            print(f"Warning: Unexpected state shape before fit: {states.shape}. Expected ({batch_size}, {board_height}, {board_width}, 5)")
            try:
                states = states.reshape((batch_size, board_height, board_width, 5))
                print(f"Reshaped states to: {states.shape}")
            except ValueError as e:
                print(f"Error reshaping states: {e}. Check feature extraction and batch assembly.")
                raise e
        targets = np.reshape(targets, (-1))
        if self.q_learning:
            self.NN.fit(states, [targets], batch_size=batch_size, epochs=1)
        else:
            probs = np.reshape(probs, (batch_size, checkersBoard.CheckersBoard.action_size))
            self.NN.fit(states, [targets, probs], batch_size=batch_size, epochs=1)

    def self_play(self, kld_threshold, num_games=1000, iterations=1, lambda_val=0.9, batch_size=1024):
        if not self.learner:
            return False
            
        for iteration in range(iterations):
            print('iteration: ' + str(iteration))
            games_left_to_play = Value('i', num_games)
            
            # Clean up existing model/session before multiprocessing
            try:
                keras.backend.clear_session()
                if self.NN is not None:
                    del self.NN
                    self.NN = None
                print("Successfully cleared previous TensorFlow session and model")
            except Exception as e:
                print(f"Warning during session/model cleanup: {e}")
                
            lock = Lock()
            game_player_pool = None
            results = []
            
            try:
                # Create the process pool
                game_player_pool = multiprocessing.Pool(
                    processes=self.game_players, 
                    initializer=self_play_init, 
                    initargs=(lock, games_left_to_play,)
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
                    # Set up TensorFlow for training
                    config = tf.ConfigProto()
                    config.gpu_options.allow_growth = True
                    self.sess = tf.Session(config=config)
                    keras.backend.set_session(self.sess)
                    
                    # Load the model for training
                    try:
                        self.NN = keras.models.load_model(self.model_filename)
                        print(f"Successfully loaded model from {self.model_filename}")
                    except Exception as e:
                        print(f"Error loading model for training: {e}")
                        continue  # Skip this iteration if we can't load the model
                    
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
        
        # Reload the model after training
        try:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            keras.backend.set_session(self.sess)
            self.NN = keras.models.load_model(self.model_filename)
        except Exception as e:
            print(f"Error reloading model after training: {e}")
            import traceback
            traceback.print_exc()

    def calculate_kld_threshold(self, current_threshold, average_game_length):
        target_game_length = 125

    @staticmethod
    def position_evaluator(evaluation_queue, evaluated_positions, model_filename):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
        model = keras.models.load_model(model_filename)
        while True:
            board, player, p_id = evaluation_queue.get()
            TDAgent.evaluate_position(model, board, player, evaluated_positions[p_id])

    @staticmethod
    def evaluate_position(model, board, current_player, evaluated_positions):
        features = TDAgent.extract_features(board, current_player)
        features = np.asarray([features])
        evaluated_positions[features.tobytes()] = model.predict(features)

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
        features = np.asarray([features])
        return self.NN.predict(features)

    def load_weights(self, filepath):
        self.NN.load_weights(filepath)

    def save_weights(self, filepath):
        self.NN.save_weights(filepath)

    def save_model(self, filepath):
        self.NN.save(filepath)
        print('neural network model saved to ' + self.model_filename)

    def load_model(self, filepath):
        self.model_filename = filepath
        self.NN.load_model(filepath)
