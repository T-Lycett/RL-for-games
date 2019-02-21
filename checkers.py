import checkersBoard as board
import random

b = board.CheckersBoard(True);
done = False
while not done:
    print('p1')
    print(b.p1_positions)
    print('p2')
    print(b.p2_positions)
    p1_moves = b.get_valid_moves(1)
    print('p1 valid moves')
    #for m in p1_moves:
        #print(m.p1_positions)
    b.set_positions(random.choice(p1_moves))
    print('p1 move')
    print(b.p1_positions)
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
#print(b.p1_kings)
#print(b.p2_kings)
