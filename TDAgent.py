import tensorflow as tf
from tensorflow import keras
import checkersBoard
import numpy as np
import math
import random
from random import shuffle
import cnn
import resNN

board_height = checkersBoard.CheckersBoard.board_height
board_width = checkersBoard.CheckersBoard.board_width

class TDAgent():
    def __init__(self, lr=0.0001, search_depth=3):
        self.search_depth = search_depth
        self.training_examples = []
        # self.NN = resNN.ResNN()
        self.NN = cnn.CNN()

    def set_lr(self, lr):
        self.NN.set_lr(lr)

    def extract_features(self, board, current_player):
        players_pieces = np.copy(board.get_player_positions(current_player))
        opp_pieces = np.copy(board.get_player_positions(-current_player))
        players_kings = np.copy(board.get_players_kings(current_player))
        opp_kings = np.copy(board.get_players_kings(-current_player))
        if current_player == -1:
            players_pieces = self.flip_pieces(players_pieces)
            opp_pieces = self.flip_pieces(opp_pieces)
            players_kings = self.flip_pieces(players_kings)
            opp_kings = self.flip_pieces(opp_kings)
        moves_until_draw = np.zeros((board_height, board_width)) + (50 - board.moves_without_capture) / 50
        return np.array([players_pieces, opp_pieces, players_kings, opp_kings, moves_until_draw])

    def flip_pieces(self, pieces):
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
        # shuffle(moves)
        for i, move in enumerate(moves):
            state, game_ended, reward = move
            state = np.asarray([state])
            losses.append([state, reward])
        states, targets = list(zip(*losses))
        states = np.asarray(states)
        targets = np.asarray(targets)
        states = np.reshape(states, newshape=(batch_size, 5, board_height, board_width))
        targets = np.reshape(targets, (-1))
        self.NN.fit_model(states, [targets], batch_size=batch_size, epochs=2)

    def self_play(self, num_games=1000, iterations=1, lambda_val=0.9, batch_size=2048, noise=0.0):
        games = 0
        e = 0.1
        for iteration in range(iterations):
            print('iteration: ' + str(iteration))
            for g in range(num_games):
                games += 1
                move_history = []
                game_ended = False
                board = checkersBoard.CheckersBoard(start_positions=True)
                current_player = 1
                state = self.extract_features(board, current_player)
                move_history.append([state, False, None])
                num_moves = 0
                while not game_ended:
                    if num_moves < 2 or np.random.rand(1) <= e:
                        move = random.choice(board.get_valid_moves(current_player))
                    else:
                        move, _ = self.get_move(board, current_player, 1, noise=noise)
                    board.set_positions(move)
                    state = self.extract_features(board, current_player)
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
                            self.training_examples.append(m)
                # if i % 20 == 0:
                    # print('game:' + str(i))
            while len(self.training_examples) >= batch_size:
                self.update_model(self.training_examples[-batch_size:], lambda_val, batch_size)
                self.training_examples[-batch_size:] = []

    def evaluate(self, board, current_player):
        features = self.extract_features(board, current_player)
        features = np.asarray([features])
        return self.NN.predict(features)

    def load_weights(self, filepath):
        self.NN.load_weights(filepath)

    def save_weights(self, filepath):
        self.NN.save_weights(filepath)

    def save_model(self, filepath):
        self.NN.save_model(filepath)

    def load_model(self, filepath):
        self.NN.load_model(filepath)
