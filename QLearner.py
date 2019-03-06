import checkersBoard
import numpy as np
import minimaxAgent
import math
import random
import matplotlib.pyplot as plt
import pickle


class QLearner():
    def __init__(self):
        self.Q = dict()

    def get_move(self, board, player):
        valid_moves = board.get_valid_moves(player)
        best_score = -math.inf
        best_action = None
        for move in valid_moves:
            state = self.get_state(move, player)
            if state in self.Q.keys():
                val = self.Q[state]
                if val > best_score:
                    best_score = val
                    best_action = move
        if best_action is not None:
            return best_action, best_score
        else:
            return random.choice(valid_moves), None

    def learn(self, q_filename, num_games=100000, iterations=100, lr=0.1, opposition_depth=2):
        wins = []
        draws = []
        losses = []
        for iteration in range(iterations):
            print('iteration: ' + str(iteration))
            wins.append(0)
            draws.append(0)
            losses.append(0)
            move_history = []
            current_player = 1
            player2 = minimaxAgent.MinimaxAgent(-1, opposition_depth)
            e = 0.1
            for g in range(num_games):
                board = checkersBoard.CheckersBoard(start_positions=True)
                move_history.append((self.get_state(board, 1), 1))
                game_ended = False
                while not game_ended:
                    if np.random.rand(1) < e:
                        p1_move = random.choice(board.get_valid_moves(1))
                    else:
                        p1_move, _ = self.get_move(board, 1)
                    board.set_positions(p1_move)
                    game_ended, winner = board.game_ended()
                    move_history.append((self.get_state(board, 1), 1))
                    if not game_ended:
                        p2_move, _ = player2.get_move(board)
                        board.set_positions(p2_move)
                        move_history.append((self.get_state(board, 1), -1))
                        game_ended, winner = board.game_ended()
                    if game_ended:
                        for i, (state, _) in enumerate(move_history):
                            if state not in self.Q.keys():
                                self.Q[state] = lr * winner
                            else:
                                self.Q[state] = self.Q[state] + lr * (winner - self.Q[state])
                        if winner == 1:
                            wins[iteration] += 1
                        elif winner == 0:
                            draws[iteration] += 1
                        elif winner == -1:
                            losses[iteration] += 1
                        move_history = []
            plt.plot(iteration + 1, wins[iteration], 'g.', iteration + 1, draws[iteration], 'b.', iteration + 1, losses[iteration], 'r.')
            plt.pause(0.001)
            self.save_q_values(q_filename)

    def play_games(self, num_games, training_examples, lr, opposition_depth, q_filename):
        pickle_rick_in = open(q_filename, 'rb')
        local_Q = pickle.load(pickle_rick_in)
        wins = 0
        draws = 0
        losses = 0
        move_history = []
        current_player = 1
        player2 = minimaxAgent.MinimaxAgent(-1, opposition_depth)
        e = 0.1
        for g in range(num_games):
            board = checkersBoard.CheckersBoard(start_positions=True)
            move_history.append((self.get_state(board, 1), 1))
            game_ended = False
            while not game_ended:
                if np.random.rand(1) < e:
                    p1_move = random.choice(board.get_valid_moves(1))
                else:
                    p1_move, _ = self.get_move(board, 1, local_Q)
                board.set_positions(p1_move)
                game_ended, winner = board.game_ended()
                move_history.append((self.get_state(board, 1), 1))
                if not game_ended:
                    p2_move, _ = player2.get_move(board)
                    board.set_positions(p2_move)
                    move_history.append((self.get_state(board, 1), -1))
                    game_ended, winner = board.game_ended()
                if game_ended:
                    for i, (state, _) in enumerate(move_history):
                        training_examples.put((state, winner))
                    if winner == 1:
                        wins += 1
                    elif winner == 0:
                        draws += 1
                    elif winner == -1:
                        losses += 1
                    move_history = []

    def get_state(self, board, current_player):
        players_pieces = np.array([[key, val] for (key, val) in board.get_players_pieces(current_player)])
        opp_pieces = np.array([[key, val] for (key, val) in board.get_players_pieces(-current_player)])
        # if current_player == -1:
            # players_pieces = self.flip_pieces(players_pieces)
            # opp_pieces = self.flip_pieces(opp_pieces)
            # players_kings = self.flip_pieces(players_kings)
            # opp_kings = self.flip_pieces(opp_kings)
        state = np.array((players_pieces, opp_pieces))
        return state.tobytes()

    def flip_pieces(self, pieces):
        pieces = pieces[::-1, :]
        pieces = pieces[:, ::-1]
        return pieces

    def save_q_values(self, filename):
        print('saving q values...')
        pickle_rick_out = open(filename, 'wb')
        pickle.dump(self.Q, pickle_rick_out)
        pickle_rick_out.close()
        print('q values saved')

    def load_q_values(self, filename):
        print('loading q values...')
        pickle_rick_in = open(filename, 'rb')
        self.Q = pickle.load(pickle_rick_in)
        print('q values loaded')

