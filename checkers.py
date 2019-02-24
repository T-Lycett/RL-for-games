import checkersBoard as board
import minimaxAgent
import random
import time
import cProfile, pstats, io
from pstats import SortKey

profile = False
if profile:
    pr =cProfile.Profile()
    pr.enable()

b = board.CheckersBoard(True);
minimax_agent = minimaxAgent.MinimaxAgent(1, 6)
minimax_agent2 = minimaxAgent.MinimaxAgent(-1, 4)
done = False
start = time.time()
while not done:
    # for m in p1_moves:
    # print(m.p1_positions)
    # b.set_positions(random.choice(p1_moves))
    p1_move, val = minimax_agent.get_move(b)
    b.set_positions(p1_move)
    print('p1 move')
    print(b.p1_positions + b.p1_kings - b.p2_positions - b.p2_kings)
    print(val)
    # time.sleep(1)
    done, _ = b.game_ended()
    if not done:
        # p2_moves = b.get_valid_moves(-1)
        # print('p2 valid moves')
        # for m in p2_moves:
        # #print(m.p2_positions)
        p2_move, val = minimax_agent2.get_move(b)
        b.set_positions(p2_move)
        # b.set_positions(random.choice(p2_moves))
        print('p2 move')
        print(b.p1_positions + b.p1_kings - b.p2_positions - b.p2_kings)
        print(val)
        done, _ = b.game_ended()
    # time.sleep(1)
    print(time.time() - start)
    # done = True
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
