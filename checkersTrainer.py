import checkersBoard as board
import minimaxAgent
import TDAgent
import random
import time
import math
import cProfile, pstats, io
from pstats import SortKey
import matplotlib
matplotlib.use('TkAgg') # Set backend *before* importing pyplot
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
import QLearner
import mcts
import torch

plt.ion() # Enable interactive mode

def set_kld_threshold(current_threshold, average_mcts_sims, target_mcts_sims):
    increment = 0.3
    margin_of_error = 0.1
    if average_mcts_sims > target_mcts_sims + (target_mcts_sims * margin_of_error):
        new_threshold = current_threshold + (current_threshold * increment)
    elif average_mcts_sims < target_mcts_sims - (target_mcts_sims * margin_of_error):
        new_threshold = current_threshold - (current_threshold * increment)
    else:
        new_threshold = current_threshold
    # new_threshold = max(new_threshold, increment)
    print('new kl-divergence threshold: ' + str(new_threshold))
    return new_threshold


if __name__ == '__main__':
    import os

    # We don't need TensorFlow-specific environment variables anymore
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    freeze_support()

    model_file = 'res64x4_dual.pth'
    num_residual_blocks = 4 # Match the likely saved model structure
    width = 64 # Match the likely saved model structure

    profile = False
    if profile:
        pr =cProfile.Profile()
        pr.enable()

    q_learning_only = False
    lr_schedule = {}
    opponent_depth = 2
    opponent_depth_increase_interval = 1
    max_opponent_depth = 8
    minimax_agent2 = minimaxAgent.MinimaxAgent(-1, 4)
    
    # Check if model file exists and create it if needed
    try:
        if not os.path.exists(model_file):
            print(f"Model file {model_file} not found. This may cause errors later.")
            
            # Create an empty model directory if needed for saving later
            model_dir = os.path.dirname(model_file)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)
                print(f"Created directory {model_dir} for model file")
                
            # You might want to create an initial model here
            # from resNN import ResNN
            # model = ResNN()
            # torch.save(model.state_dict(), model_file)
            # print(f"Created initial model file {model_file}")
    except Exception as e:
        print(f"Error checking/creating model path: {e}")
    
    # Initialize TDAgent with error handling
    try:
        # Use device-aware initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        TD_agent = TDAgent.TDAgent(lr=0.00001, model_filename=model_file, q_learning=q_learning_only, width=width, residual_blocks=num_residual_blocks)
    except Exception as e:
        print(f"Error initializing TDAgent with model {model_file}: {e}")
        import traceback
        traceback.print_exc()
        print("Exiting program due to critical initialization error.")
        import sys
        sys.exit(1)
        
    q_learner = QLearner.QLearner()
    start = time.time()
    iterations = 1000
    self_play_games_per_iteration = 50
    test_games = 10
    epoch_per_iteration = 1
    kld_threshold = 0.00288
    target_average_num_sims = 200
    calibration_runs = 1
    wins = []
    draws = []
    losses = []
    q_file = './q_values8x6.pickle'
    # q_learner.load_q_values(q_file)
    
    # Create figure and axes outside the loop
    fig, ax = plt.subplots()
    plt.show(block=False) # Show the window initially non-blocking
    plt.pause(0.1) # Add a small pause to allow window draw
    
    for i in range(iterations):
        winner = None
        searches = []
        if i in lr_schedule:
            TD_agent.set_lr(lr_schedule[i])
        print('iteration: ' + str(i))
        if i >= calibration_runs:
            TD_agent.self_play(kld_threshold, num_games=self_play_games_per_iteration, iterations=epoch_per_iteration)
        # TD_agent.save_model(model_file)
        # q_learner.learn(q_file, num_games=1000, iterations=20, opposition_depth=opponent_depth)
        # q_learner.save_q_values(q_file)
        if i >= 0:
            wins.append(0)
            draws.append(0)
            losses.append(0)
            for g in range(test_games):
                print('Game: ' + str(g + 1))
                num_moves = 0
                done = False
                minimax_agent = minimaxAgent.MinimaxAgent(-1, opponent_depth)
                mcts_instance = mcts.MCTS(TD_agent.NN, use_policy_head=not q_learning_only)
                b = board.CheckersBoard(True)
                while not done:
                    num_moves += 1
                    # for m in p1_moves:
                    # print(m.p1_positions)
                    # b.set_positions(random.choice(p1_moves))
                    if b.current_player == 1:
                        print('game ' + str(g + 1) + ' - move ' + str(num_moves) + ' - player 1')
                        move_start_time = time.time()
                        try:
                            move, val = TD_agent.get_move(b, 1, mcts_instance, kld_threshold)
                            move_end_time = time.time() - move_start_time
                            print('elapsed time: ' + str(move_end_time))
                            if mcts_instance.mcts_sims != 0:
                                print('max depth: ' + str(mcts_instance.max_depth))
                                print('number of simulations: ' + str(mcts_instance.mcts_sims))
                                if move_end_time != 0:
                                    print('nodes per second: ' + str(mcts_instance.mcts_sims / move_end_time))
                                searches.append(mcts_instance.mcts_sims)
                        except Exception as e:
                            print(f"ERROR in TD_agent.get_move: {e}")
                            import traceback
                            traceback.print_exc()
                            move, val = None, None
                    else:
                        print('game ' + str(g + 1) + ' - move ' + str(num_moves) + ' - player 2')
                        move_start_time = time.time()
                        try:
                            move, val = minimax_agent.get_move(b)
                            move_end_time = time.time() - move_start_time
                            print('elapsed time: ' + str(move_end_time))
                            if move_end_time != 0:
                                print('nodes per second: ' + str(minimax_agent.nodes_visited / move_end_time))
                        except Exception as e:
                            print(f"ERROR in minimax_agent.get_move: {e}")
                            import traceback
                            traceback.print_exc()
                            move, val = None, None
                    # p1_move, val = q_learner.get_move(b, 1)
                    try:
                        if move is None:
                            print('Error: valid move list was empty, resetting the game.')
                            b = board.CheckersBoard(True)
                        else:
                            b.set_positions(move)
                    except Exception as e:
                        print(f"ERROR setting board positions: {e}")
                        import traceback
                        traceback.print_exc()
                        print('Resetting the game due to board error.')
                        b = board.CheckersBoard(True)

                    print(b.p1_positions + b.p1_kings - b.p2_positions - b.p2_kings)
                    print('moves until draw: ' + str(50 - b.moves_without_capture))
                    print(str(val))
                    print('')
                    # time.sleep(5)
                    done, winner = b.game_ended()
                if winner == 1:
                    wins[int(i/1)] += 1
                elif winner == 0:
                    draws[int(i/1)] += 1
                elif winner == -1:
                    losses[int(i/1)] += 1
                
                # Use axes object for plotting
                ax.cla() 
                ax.plot(range(1, 1 + len(wins)), wins, 'g.', label='Wins')
                ax.plot(range(1, 1 + len(draws)), draws, 'b.', label='Draws')
                ax.plot(range(1, 1 + len(losses)), losses, 'r.', label='Losses')
                ax.legend()
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Count")
                ax.set_title("Training Progress")
                
                # Explicitly draw and pause
                plt.draw() 
                plt.pause(0.1)

            average_sims = sum(searches) / len(searches)
            print('average sims: ' + str(average_sims))
            print('min sims: ' + str(min(searches)))
            print('max sims: ' + str(max(searches)))
            print('kl-divergence threshold: ' + str(kld_threshold))
            kld_threshold = set_kld_threshold(kld_threshold, average_sims, target_average_num_sims)
                # time.sleep(1)
        print('opponent depth: ' + str(opponent_depth))
        print('wins: ' + str(wins))
        print('draws: ' + str(draws))
        print('losses: ' + str(losses))
        if losses[int(i)] == 0:
            opponent_depth += opponent_depth_increase_interval
            opponent_depth = min(max_opponent_depth, opponent_depth)
        print('average time per iteration: ' + str((time.time() - start) / (i + 1)))
    
    # plt.show() # Commented out - plt.ion() keeps the plot interactive
    
    if profile:
        # profiling code from: https://docs.python.org/3/library/profile.html#module-profile
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
