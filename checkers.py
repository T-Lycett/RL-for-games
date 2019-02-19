import checkersBoard as board

b = board.CheckersBoard(True);
print(b.p1_positions)
print(b.p2_positions)
print('valid moves')
for move in b.getValidMoves(1):
    print(move.p1_positions)
#print(b.p1_kings)
#print(b.p2_kings)
