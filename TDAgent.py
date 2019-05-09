import tensorflow as tf
from tensorflow import keras
import checkersBoard
import numpy as np
import math
from random import shuffle
import multiprocessing
import mcts
from multiprocessing import Lock, Value

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
    return np.array([players_pieces, opp_pieces, players_kings, opp_kings, moves_until_draw])


def get_move(board, player, mcts_instance, kld_threshold, temperature, max_sims=None):
    moves = board.get_valid_moves(player, include_index=True, include_chain_jumps=False)
    num_moves = len(moves)
    if num_moves == 0:
        return None, None
    if num_moves == 1:
        mcts_instance.mcts_sims = 0
        probs = np.zeros((checkersBoard.CheckersBoard.action_size))
        index = int(moves[0][1])
        probs[index] = 1
        if sum(probs) > 1.01:
            print(probs)
        return moves[0][0], probs
    else:
        probs = mcts_instance.get_probabilities(board, player, kld_threshold=kld_threshold, max_sims=max_sims, temperature=temperature)
        choices = np.ndarray((checkersBoard.CheckersBoard.action_size), dtype=checkersBoard.CheckersBoard)
        for m, i in moves:
            choices[int(i)] = m
        move_index = np.random.multinomial(1, probs)
        move_index = np.where(move_index == 1)[0][0]
        chosen_move = choices[move_index]
        return chosen_move, probs


def self_play_init(l, val):
    global self_play_lock
    self_play_lock = l
    global games_to_play
    games_to_play = val


def self_play_game_player(model_filename, kld_threshold):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
    model = keras.models.load_model(model_filename)
    training_examples = []
    games = 0
    while True:
        with self_play_lock:
            print('games left: ' + str(games_to_play.value))
            if games_to_play.value == 0:
                break
            else:
                games_to_play.value -= 1
        mcts_instance = mcts.MCTS(model)
        games += 1
        move_history = []
        game_ended = False
        board = checkersBoard.CheckersBoard(start_positions=True)
        current_player = 1
        num_moves = 0
        moves_until_t0 = math.inf
        num_mcts_sims = []
        while not game_ended:
            if num_moves < moves_until_t0:
                move, probs = get_move(board, current_player, mcts_instance, kld_threshold=kld_threshold, temperature=0.5)
            else:
                move, probs = get_move(board, current_player, mcts_instance, kld_threshold, temperature=0)
            state = extract_features(board, current_player)
            assert not np.all(np.isnan(probs))
            move_history.append([state, current_player, None, probs])
            if mcts_instance.mcts_sims != 0:
                num_mcts_sims.append(mcts_instance.mcts_sims)
            if sum(probs) > 1.01:
                print('error: sum of probabilities is more than 1: ' + str(sum(probs)))
            elif sum(probs) < 0.99:
                print('error: sum of probabilities is less than 1: ' + str(sum(probs)))
            board.set_positions(move)
            game_ended, winner = board.game_ended()
            current_player = board.current_player
            num_moves += 1
            if game_ended:
                if winner == 1:
                    winner_str = 'player 1'
                elif winner == -1:
                    winner_str = 'player 2'
                else:
                    winner_str = 'draw'
                print('finished game, temp = 0 at move ' + str(moves_until_t0) + ', game length: ' + str(num_moves) + ', average mcts sims: ' + str(sum(num_mcts_sims) / len(num_mcts_sims)) + ', max mcts sims: ' + str(max(num_mcts_sims)) + ', winner: ' + winner_str)
                state = extract_features(board, current_player)
                assert not np.all(np.isnan(probs))
                move_history.append([state, current_player, None, probs])
                for i, (_, cur_player, r, _) in enumerate(move_history):
                    move_history[i][2] = cur_player * winner
                for m in move_history:
                    training_examples.append(m)
    keras.backend.clear_session()
    del model
    return training_examples


class TDAgent:
    def __init__(self, model_filename, learner=True, lr=0.0001, search_depth=3):
        self.learner = learner
        self.lr = lr
        self.search_depth = search_depth
        self.training_examples = []
        self.model_filename = model_filename

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        keras.backend.set_session(self.sess)
        self.NN = keras.models.load_model(model_filename)
        if self.learner:
            self.game_players = 5  # int(multiprocessing.cpu_count() / 2)

    def set_lr(self, lr):
        self.lr = lr

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
        return np.array([players_pieces, opp_pieces, players_kings, opp_kings, moves_until_draw])

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
            nn_eval = self.evaluate(moves[0][0], moves[0][0].current_player)
            if player != moves[0][0].current_player:
                nn_eval *= -1
            return moves[0][0], nn_eval
        else:
            probs = mcts_instance.get_probabilities(board, player, kld_threshold=kld_threshold, temperature=0, verbose=True)
            choices = np.ndarray(checkersBoard.CheckersBoard.action_size, dtype=checkersBoard.CheckersBoard)
            for move, i in moves:
                choices[int(i)] = move
            move_index = np.random.multinomial(1, probs)
            move_index = np.where(move_index == 1)[0][0]
            chosen_move = choices[move_index]

            current_state = self.extract_features(board, board.current_player).tobytes()
            move_state = self.extract_features(chosen_move, chosen_move.current_player).tobytes()
            nn_val = self.evaluate(chosen_move, chosen_move.current_player)
            mcts_val = mcts_instance.Qsa[(current_state, move_state)]
            if player != chosen_move.current_player:
                nn_val *= -1
            eval_str = 'Neural Network: ' + str(nn_val[0][0]) + ', MCTS: ' + str(mcts_val)
            return chosen_move, eval_str

    def update_model(self, moves, batch_size):
        print('updating neural network...')
        losses = []
        for i, move in enumerate(moves):
            state, _, reward, probs = move
            state = np.asarray([state])
            losses.append([state, reward, probs])
        states, targets, probs = list(zip(*losses))
        assert not np.all(np.isnan(probs))
        states = np.asarray(states)
        targets = np.asarray(targets)
        states = np.reshape(states, newshape=(batch_size, 5, board_height, board_width))
        targets = np.reshape(targets, (-1))
        self.NN.fit(states, [targets], batch_size=batch_size, epochs=1)

    def self_play(self, kld_threshold, num_games=1000, iterations=1, batch_size=1024):
        if not self.learner:
            return False
        for iteration in range(iterations):
            print('iteration: ' + str(iteration))
            games_left_to_play = Value('i', num_games)
            keras.backend.clear_session()
            if self.NN is not None:
                del self.NN
                self.NN = None
            lock = Lock()
            game_player_pool = multiprocessing.Pool(processes=self.game_players, initializer=self_play_init, initargs=(lock, games_left_to_play,))
            results = [game_player_pool.apply_async(self_play_game_player, args=(self.model_filename, kld_threshold)) for p in range(self.game_players)]
            game_player_pool.close()
            game_player_pool.join()
            self_play_results = [r.get() for r in results]
            for sublist in self_play_results:
                for example in sublist:
                    self.training_examples.append(example)
            print('training examples: ' + str(len(self.training_examples)))
            if len(self.training_examples) > batch_size:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)
                keras.backend.set_session(self.sess)
                self.NN = keras.models.load_model(self.model_filename)
                self.training_examples = self.deduplicate_training_data(self.training_examples)
                shuffle(self.training_examples)
                while len(self.training_examples) > batch_size:
                    try:
                        self.update_model(self.training_examples[-batch_size:], batch_size)
                        self.training_examples[-batch_size:] = []
                    except:
                        print('possible gpu error')
                self.save_model(self.model_filename)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        keras.backend.set_session(self.sess)
        self.NN = keras.models.load_model(self.model_filename)

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

    def load_model(self, filepath):
        self.model_filename = filepath
        self.NN.load_model(filepath)
