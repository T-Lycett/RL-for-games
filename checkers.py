import checkersBoard as board
import minimaxAgent
import TDAgent
import random
import time
import cProfile, pstats, io
from pstats import SortKey
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
import QLearner
import mcts

if __name__ == '__main__':
    freeze_support()

    model_file = 'res128x5.h5'

    profile = False
    if profile:
        pr =cProfile.Profile()
        pr.enable()

    lr_schedule = {0: 0.0000001}
    opponent_depth = 2
    minimax_agent2 = minimaxAgent.MinimaxAgent(-1, 4)
    TD_agent = TDAgent.TDAgent(lr=0.00001, model_filename=model_file)
    # TD_agent.load_weights('./weights/res_nn_Model')
    # TD_agent.load_model('cnn64x2.h5')
    q_learner = QLearner.QLearner()
    start = time.time()
    iterations = 100
    test_games = 10
    wins = []
    draws = []
    losses = []
    q_file = './q_values8x6.pickle'
    # q_learner.load_q_values(q_file)
    for i in range(iterations):
        winner = None
        # if i in lr_schedule:
            # TD_agent.set_lr(lr_schedule[i])
        print('iteration: ' + str(i))
        TD_agent.self_play(num_games=100)
        TD_agent.save_model(model_file)
        # q_learner.learn(q_file, num_games=1000, iterations=20, opposition_depth=opponent_depth)
        # q_learner.save_q_values(q_file)
        if i % 1 != 0:
            continue
        wins.append(0)
        draws.append(0)
        losses.append(0)
        for g in range(test_games):
            done = False
            minimax_agent = minimaxAgent.MinimaxAgent(-1, opponent_depth)
            mcts_instance = mcts.MCTS(TD_agent.NN)
            b = board.CheckersBoard(True)
            while not done:
                # for m in p1_moves:
                # print(m.p1_positions)
                # b.set_positions(random.choice(p1_moves))
                if b.current_player == 1:
                    print('player 1 move')
                    move_start_time = time.time()
                    move, val = TD_agent.get_move(b, 1, mcts_instance)
                    move_end_tine = time.time() - move_start_time
                    print(move_end_tine)
                else:
                    print('player 2 move')
                    move_start_time = time.time()
                    move, val = minimax_agent.get_move(b)
                    move_end_tine = time.time() - move_start_time
                    print(move_end_tine)
                    if move_end_tine != 0:
                        print('nodes per second: ' + str(minimax_agent.nodes_visited / move_end_tine))
                # p1_move, val = q_learner.get_move(b, 1)
                try:
                    b.set_positions(move)
                except:
                    if move is None:
                        print('Error: valid move list was empty, resetting the game.')
                        b = board.CheckersBoard(True)

                print(b.p1_positions + b.p1_kings - b.p2_positions - b.p2_kings)
                print(val)
                # time.sleep(5)
                done, winner = b.game_ended()
            if winner == 1:
                wins[int(i/1)] += 1
            elif winner == 0:
                draws[int(i/1)] += 1
            elif winner == -1:
                losses[int(i/1)] += 1
        # plt.cla()
        plt.plot(i + 1, wins[i], 'g.', i + 1, draws[i], 'b.', i + 1, losses[i], 'r.')
        plt.pause(0.001)

                # time.sleep(1)
        print('opponent depth: ' + str(opponent_depth))
        print('wins: ' + str(wins))
        print('draws: ' + str(draws))
        print('losses: ' + str(losses))
        if losses[int(i)] == 0:
            opponent_depth += 1
        print((time.time() - start) / (i + 1))
        time.sleep(5)
    plt.show()
    if profile:
        # profiling code from: https://docs.python.org/3/library/profile.html#module-profile
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

#print(b.p1_kings)
#print(b.p2_kings)


