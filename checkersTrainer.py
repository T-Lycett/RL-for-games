import checkersBoard as board
import minimaxAgent
import TDAgent
import time
import cProfile, pstats, io
from pstats import SortKey
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
import mcts


def set_kld_threshold(current_threshold, average_mcts_sims, target_mcts_sims):
    increment = 0.2
    margin_of_error = 0.05
    if average_mcts_sims > target_mcts_sims + (target_mcts_sims * margin_of_error):
        new_threshold = current_threshold + (current_threshold * increment)
    elif average_mcts_sims < target_mcts_sims - (target_mcts_sims * margin_of_error):
        new_threshold = current_threshold - (current_threshold * increment)
    else:
        new_threshold = current_threshold
    print('new kl-divergence threshold: ' + str(new_threshold))
    return new_threshold


if __name__ == '__main__':
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    freeze_support()

    model_file = 'res128x5T01.h5'

    profile = False
    if profile:
        pr =cProfile.Profile()
        pr.enable()

    opponent_depth = 2
    minimax_agent2 = minimaxAgent.MinimaxAgent(-1, 4)
    TD_agent = TDAgent.TDAgent(lr=0.00001, model_filename=model_file)

    iterations = 15
    test_games = 10
    kld_threshold = 0.01
    target_average_num_sims = 120

    wins = []
    draws = []
    losses = []
    start = time.time()

    for i in range(iterations):
        winner = None
        searches = []

        print('iteration: ' + str(i))
        if i >= 1:
            TD_agent.self_play(kld_threshold, num_games=100, iterations=2)

        if i >= 0:
            wins.append(0)
            draws.append(0)
            losses.append(0)
            for g in range(test_games):
                print('Game: ' + str(g + 1))
                num_moves = 0
                done = False
                minimax_agent = minimaxAgent.MinimaxAgent(-1, opponent_depth)
                mcts_instance = mcts.MCTS(TD_agent.NN)
                b = board.CheckersBoard(True)
                while not done:
                    num_moves += 1
                    if b.current_player == 1:
                        print('move ' + str(num_moves) + ' - player 1')
                        move_start_time = time.time()
                        move, val = TD_agent.get_move(b, 1, mcts_instance, kld_threshold)
                        move_end_time = time.time() - move_start_time
                        print('elapsed time: ' + str(move_end_time))
                        if mcts_instance.mcts_sims != 0:
                            print('number of simulations: ' + str(mcts_instance.mcts_sims))
                            if move_end_time != 0:
                                print('nodes per second: ' + str(mcts_instance.mcts_sims / move_end_time))
                            searches.append(mcts_instance.mcts_sims)
                    else:
                        print('move ' + str(num_moves) + ' - player 2')
                        move_start_time = time.time()
                        move, val = minimax_agent.get_move(b)
                        move_end_time = time.time() - move_start_time
                        print('elapsed time: ' + str(move_end_time))
                        if move_end_time != 0:
                            print('nodes per second: ' + str(minimax_agent.nodes_visited / move_end_time))
                    try:
                        b.set_positions(move)
                    except:
                        if move is None:
                            print('Error: valid move list was empty, resetting the game.')
                            b = board.CheckersBoard(True)

                    print(b.p1_positions + b.p1_kings - b.p2_positions - b.p2_kings)
                    print('moves until draw: ' + str(50 - b.moves_without_capture))
                    print(str(val))
                    print('')
                    done, winner = b.game_ended()
                if winner == 1:
                    wins[int(i/1)] += 1
                elif winner == 0:
                    draws[int(i/1)] += 1
                elif winner == -1:
                    losses[int(i/1)] += 1
            average_sims = sum(searches) / len(searches)
            print('average sims: ' + str(average_sims))
            print('min sims: ' + str(min(searches)))
            print('max sims: ' + str(max(searches)))
            print('kl-divergence threshold: ' + str(kld_threshold))
            kld_threshold = set_kld_threshold(kld_threshold, average_sims, target_average_num_sims)

        plt.plot(i + 1, wins[i], 'g.', i + 1, draws[i], 'b.', i + 1, losses[i], 'r.')
        plt.pause(0.001)

        print('opponent depth: ' + str(opponent_depth))
        print('wins: ' + str(wins))
        print('draws: ' + str(draws))
        print('losses: ' + str(losses))
        if losses[int(i)] == 0:
            opponent_depth += 1
            opponent_depth = min(5, opponent_depth)
        print('average time per iteration: ' + str((time.time() - start) / (i + 1)))
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


