import checkersBoard as board
import minimaxAgent
import random
import time

b = board.CheckersBoard(True);
minimax_agent = minimaxAgent.MinimaxAgent(1)
done = False
while not done:
    p1_moves = b.get_valid_moves(1)
    #for m in p1_moves:
        #print(m.p1_positions)
    #b.set_positions(random.choice(p1_moves))
    p1_move, _ = minimax_agent.minimax(b)
    b.set_positions(p1_move)
    print('p1 move')
    print(b.p1_positions)
    time.sleep(1)
    done, _ = b.game_ended()
    if not done:
        p2_moves = b.get_valid_moves(-1)
        print('p2 valid moves')
        #for m in p2_moves:
        # #print(m.p2_positions)
        b.set_positions(random.choice(p2_moves))
        print('p2 move')
        print(b.p2_positions)
        done, _ = b.game_ended()
    time.sleep(1)
#print(b.p1_kings)
#print(b.p2_kings)
