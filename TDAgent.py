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

board_height = checkersBoard.CheckersBoard.board_height
board_width = checkersBoard.CheckersBoard.board_width


def evaluate(model, board, current_player):
    features = extract_features(board, current_player)
    features = np.asarray([features])
    return model.predict(features)


def minimax(model, board, player, max_depth, current_depth=0, alpha=-math.inf, beta=math.inf, noise=0.0):
    if current_depth % 2 == 0:
        current_player = player
    else:
        current_player = -player
    game_ended, _ = board.game_ended()
    if current_depth >= max_depth or game_ended:
        return board, evaluate(model, board, -current_player)
    actions = board.get_valid_moves(current_player)
    shuffle(actions)
    if current_player == player:
        best_score = -math.inf
        best_action_index = -1
        for i, a in enumerate(actions):
            _, score = minimax(model, board=a, player=player, max_depth=max_depth, current_depth=current_depth + 1, alpha=alpha, beta=beta)
            if noise and current_depth == 0:
                score += random.uniform(-noise, noise)
            if score > best_score:
                best_score = score
                best_action_index = i
                alpha = best_score
            if alpha >= beta:
                break
        return actions[best_action_index], best_score
    else:
        best_score = math.inf
        best_action_index = -1
        for i, a in enumerate(actions):
            _, score = minimax(model, board=a, player=player, max_depth=max_depth, current_depth=current_depth + 1, alpha=alpha, beta=beta)
            if noise and current_depth == 0:
                score += random.uniform(-noise, noise)
            if score < best_score:
                best_score = score
                best_action_index = i
                beta = best_score
            if alpha >= beta:
                break
        return actions[best_action_index], best_score


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


def get_move(model, board, player, search_depth, noise=0.0):
    moves = board.get_valid_moves(player)
    num_moves = len(moves)
    if num_moves == 0:
        return None, None
    if num_moves == 1:
        return moves[0], evaluate(model, moves[0], player)
    else:
        return minimax(model, board, player, search_depth, noise=noise)


def self_play_game_player(model_filename, num_games=1000, noise=0.0):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
    model = keras.models.load_model(model_filename)
    training_examples = []
    games = 0
    e = 0.01
    for g in range(num_games):
        games += 1
        move_history = []
        game_ended = False
        board = checkersBoard.CheckersBoard(start_positions=True)
        current_player = 1
        state = extract_features(board, current_player)
        move_history.append([state, False, None])
        num_moves = 0
        while not game_ended:
            if np.random.rand(1) <= e or num_moves < 1:
                move = random.choice(board.get_valid_moves(current_player))
            else:
                move, _ = get_move(model, board, current_player, 1, noise=noise)
            board.set_positions(move)
            state = extract_features(board, current_player)
            game_ended, reward = board.game_ended()
            if reward is not None:
                reward *= current_player
            move_history.append([state, game_ended, reward])
            current_player *= -1
            num_moves += 1
            # if num_moves % update_frequency == 0 or game_ended:
            if game_ended:
                step = len(move_history) - 1
                for _, _, r in reversed(move_history):
                    if step > 0:
                        move_history[step - 1][2] = -r
                        step -= 1
                for m in move_history:
                    training_examples.append(m)
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

    def get_move(self, board, player, search_depth, noise=0.0):
        moves = board.get_valid_moves(player)
        num_moves = len(moves)
        if num_moves == 0:
            return None, None
        if num_moves == 1:
            return moves[0], self.evaluate(moves[0], player)
        else:
            return self.minimax(board, player, search_depth, noise=noise)

    def minimax(self, board, player, max_depth, current_depth=0, alpha=-math.inf, beta=math.inf, noise=0.0):
        if current_depth % 2 == 0:
            current_player = player
        else:
            current_player = -player
        game_ended, _ = board.game_ended()
        if current_depth >= max_depth or game_ended:
            return board, self.evaluate(board, -current_player)
        actions = board.get_valid_moves(current_player)
        shuffle(actions)
        if current_player == player:
            best_score = -math.inf
            best_action_index = -1
            for i, a in enumerate(actions):
                _, score = self.minimax(a, player, max_depth, current_depth + 1, alpha, beta)
                if noise != 0 and current_depth == 0:
                    score += random.uniform(-noise, noise)
                if score > best_score:
                    best_score = score
                    best_action_index = i
                    alpha = best_score
                if alpha >= beta:
                    break
            return actions[best_action_index], best_score
        else:
            best_score = math.inf
            best_action_index = -1
            for i, a in enumerate(actions):
                _, score = self.minimax(a, player, max_depth, current_depth + 1, alpha, beta)
                if noise != 0 and current_depth == 0:
                    score += random.uniform(-noise, noise)
                if score < best_score:
                    best_score = score
                    best_action_index = i
                    beta = best_score
                if alpha >= beta:
                    break
            return actions[best_action_index], best_score

    def update_model(self, moves, lambda_val, batch_size):
        losses = []
        shuffle(moves)
        for i, move in enumerate(moves):
            state, game_ended, reward = move
            state = np.asarray([state])
            losses.append([state, reward])
        states, targets = list(zip(*losses))
        states = np.asarray(states)
        targets = np.asarray(targets)
        states = np.reshape(states, newshape=(batch_size, 5, board_height, board_width))
        targets = np.reshape(targets, (-1))
        self.NN.fit(states, [targets], batch_size=batch_size, epochs=2)

    def self_play(self, num_games=1000, iterations=1, lambda_val=0.9, batch_size=4096, noise=0.05):
        if not self.learner:
            return False
        games = 0
        e = 0.05
        for iteration in range(iterations):
            print('iteration: ' + str(iteration))
            # self.evaluation_queue = multiprocessing.Manager().Queue()
            # self.evaluated_positions = [multiprocessing.Manager().dict() for _ in range(self.game_players)]
            # self.pos_eval_worker = multiprocessing.Process(target=self.position_evaluator, args=(self.evaluation_queue, self.evaluated_positions, self.model_filename))
            # self.pos_eval_worker.start()
            game_player_pool = multiprocessing.Pool(processes=self.game_players)
            results = [game_player_pool.apply_async(self_play_game_player, args=(self.model_filename, int(num_games / self.game_players), noise)) for p in range(self.game_players)]
            game_player_pool.close()
            game_player_pool.join()
            self.training_examples = [r.get() for r in results]
            self.training_examples = [item for sublist in self.training_examples for item in sublist]
            print('training examples: ' + str(len(self.training_examples)))
            # self.pos_eval_worker.close()
            while len(self.training_examples) >= batch_size:
                self.update_model(self.training_examples[-batch_size:], lambda_val, batch_size)
                self.training_examples[-batch_size:] = []
            self.save_model(self.model_filename)

    @staticmethod
    def position_evaluator(evaluation_queue, evaluated_positions, model_filename):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        # (nothing gets printed in Jupyter, only if you run it standalone)
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
