import tensorflow as tf
from tensorflow import keras
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


def get_move(board, player, mcts_instance, temperature):
    moves = board.get_valid_moves(player, include_index=True, include_chain_jumps=False)
    num_moves = len(moves)
    if num_moves == 0:
        return None, None
    if num_moves == 1:
        probs = np.zeros((checkersBoard.CheckersBoard.action_size))
        index = int(moves[0][1])
        probs[index] = 1
        if sum(probs) > 1.01:
            print(probs)
        return moves[0][0], probs
    else:
        probs = mcts_instance.get_probabilities(board, player, num_sims=60, temperature=temperature)
        choices = np.ndarray((checkersBoard.CheckersBoard.action_size), dtype=checkersBoard.CheckersBoard)
        for m, i in moves:
            choices[int(i)] = m
        move_index = np.random.multinomial(1, probs)
        move_index = np.where(move_index == 1)[0][0]
        chosen_move = choices[move_index]
        return chosen_move, probs


def self_play_game_player(model_filename, num_games=1000, noise=0.0):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
    model = keras.models.load_model(model_filename)
    mcts_instance = mcts.MCTS(model)
    training_examples = []
    games = 0
    for g in range(num_games):
        games += 1
        move_history = []
        game_ended = False
        board = checkersBoard.CheckersBoard(start_positions=True)
        current_player = 1
        # state = extract_features(board, current_player)
        # move_history.append([state, False, None, None])
        num_moves = 0
        moves_until_t0 = 30
        while not game_ended:
            if num_moves < moves_until_t0:
                move, probs = get_move(board, current_player, mcts_instance, 0.5)
            else:
                move, probs = get_move(board, current_player, mcts_instance, 0)
            if sum(probs) > 1.01:
                print('error: sum of probabilities is more than 1: ' + str(sum(probs)))
            elif sum(probs) < 0.99:
                print('error: sum of probabilities is less than 1: ' + str(sum(probs)))
            state = extract_features(board, current_player)
            assert not np.all(np.isnan(probs))
            move_history.append([state, current_player, None, probs])
            board.set_positions(move)
            game_ended, winner = board.game_ended()
            current_player = board.current_player
            num_moves += 1
            # if num_moves % update_frequency == 0 or game_ended:
            if game_ended:
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


class TDAgent():
    def __init__(self, model_filename, learner=True, lr=0.0001, search_depth=3):
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
            self.game_players = int(multiprocessing.cpu_count() / 2)

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

    def get_move(self, board, player, mcts_instance):
        moves = board.get_valid_moves(player, include_index=True, include_chain_jumps=False)
        num_moves = len(moves)
        if num_moves == 0:
            return None, None
        if num_moves == 1:
            return moves[0][0], self.evaluate(moves[0][0], player)
        else:
            probs = mcts_instance.get_probabilities(board, player, num_sims=60, temperature=0, verbose=True)
            choices = np.ndarray(checkersBoard.CheckersBoard.action_size, dtype=checkersBoard.CheckersBoard)
            for move, i in moves:
                choices[int(i)] = move
            move = choices[weighted_pick(probs)]
            return move, self.evaluate(move, player)

    def update_model(self, moves, lambda_val, batch_size):
        losses = []
        shuffle(moves)
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
        # probs = np.reshape(probs, (batch_size, checkersBoard.CheckersBoard.action_size))
        # self.NN.fit(states, [targets, probs], batch_size=batch_size, epochs=1)
        self.NN.fit(states, [targets], batch_size=batch_size, epochs=1)

    def self_play(self, num_games=1000, iterations=1, lambda_val=0.9, batch_size=1024, noise=0.05):
        if not self.learner:
            return False
        for iteration in range(iterations):
            print('iteration: ' + str(iteration))
            # self.evaluation_queue = multiprocessing.Manager().Queue()
            # self.evaluated_positions = [multiprocessing.Manager().dict() for _ in range(self.game_players)]
            # self.pos_eval_worker = multiprocessing.Process(target=self.position_evaluator, args=(self.evaluation_queue, self.evaluated_positions, self.model_filename))
            # self.pos_eval_worker.start()
            keras.backend.clear_session()
            del self.NN
            game_player_pool = multiprocessing.Pool(processes=self.game_players)
            results = [game_player_pool.apply_async(self_play_game_player, args=(self.model_filename, int(num_games / self.game_players), noise)) for p in range(self.game_players)]
            game_player_pool.close()
            game_player_pool.join()
            self.training_examples = [r.get() for r in results]
            self.training_examples = [item for sublist in self.training_examples for item in sublist]
            print('training examples: ' + str(len(self.training_examples)))
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
            self.sess = tf.Session(config=config)
            keras.backend.set_session(self.sess)  # set this TensorFlow session as the default session for Keras
            self.NN = keras.models.load_model(self.model_filename)
            # self.NN.compile(keras.optimizers.Adam(lr=self.lr), loss=tf.losses.mean_squared_error)
            # self.pos_eval_worker.close()
            self.training_examples = self.deduplicate_training_data(self.training_examples)
            while len(self.training_examples) > batch_size:
                self.update_model(self.training_examples[-batch_size:], lambda_val, batch_size)
                self.training_examples[-batch_size:] = []
            # self.save_model(self.model_filename)

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

    def load_model(self, filepath):
        self.model_filename = filepath
        self.NN.load_model(filepath)
