import numpy as np
from enum import Enum


class Direction(Enum):
    UP_LEFT = 0
    UP_RIGHT = 1
    DOWN_LEFT = 2
    DOWN_RIGHT = 3


class CheckersBoard:

    board_height = 8
    board_width = 8
    action_size = 256

    def __init__(self, start_positions=False, board=None):
        self.reset()
        if start_positions:
            self.set_start_positions()
        if board is not None:
            self.set_positions(board)

    def reset(self):
        self.p1_positions = np.zeros((CheckersBoard.board_height, CheckersBoard.board_width))
        self.p2_positions = np.zeros((CheckersBoard.board_height, CheckersBoard.board_width))
        self.p1_kings = np.zeros((CheckersBoard.board_height, CheckersBoard.board_width))
        self.p2_kings = np.zeros((CheckersBoard.board_height, CheckersBoard.board_width))
        self.moves_without_capture = 0
        self.p1_valid_moves = []
        self.p2_valid_moves = []
        self.p1_valid_moves_single_jumps = []
        self.p2_valid_moves_single_jumps = []
        self.p1_valid_moves_updated = False
        self.p2_valid_moves_updated = False
        self.p1_pieces = {}
        self.p2_pieces = {}
        self.current_player = 1
        self.chain_jump = False
        self.chain_jump_from = None

    def set_start_positions(self):
        self.reset()
        self.p1_positions[7, ::2] = 1
        self.p1_positions[6, :] = 1
        self.p1_positions[6, ::2] = 0
        self.p1_positions[5, ::2] = 1
        self.p2_positions[0, :] = 1
        self.p2_positions[0, ::2] = 0
        self.p2_positions[1, ::2] = 1
        self.p2_positions[2, :] = 1
        self.p2_positions[2, ::2] = 0
        for x in range(0, CheckersBoard.board_width, 2):
            self.p1_pieces[(7, x)] = 0
            self.p1_pieces[(5, x)] = 0
            self.p2_pieces[(1, x)] = 0
        for x in range(1, CheckersBoard.board_width, 2):
            self.p1_pieces[(6, x)] = 0
            self.p2_pieces[(0, x)] = 0
            self.p2_pieces[(2, x)] = 0
        self.current_player = 1
        self.chain_jump = False
        self.chain_jump_from = None
        self.update_valid_moves(1)
        self.update_valid_moves(-1)

    def set_positions(self, board):
        self.p1_positions = board.p1_positions.copy()
        self.p2_positions = board.p2_positions.copy()
        self.p1_kings = board.p1_kings.copy()
        self.p2_kings = board.p2_kings.copy()
        self.p1_valid_moves = board.p1_valid_moves.copy()
        self.p2_valid_moves = board.p2_valid_moves.copy()
        self.p1_valid_moves_single_jumps = board.p1_valid_moves_single_jumps.copy()
        self.p2_valid_moves_single_jumps = board.p2_valid_moves_single_jumps.copy()
        self.p1_pieces = board.p1_pieces.copy()
        self.p2_pieces = board.p2_pieces.copy()
        self.moves_without_capture = board.moves_without_capture
        self.p1_valid_moves_updated = board.p1_valid_moves_updated
        self.p2_valid_moves_updated = board.p2_valid_moves_updated
        self.current_player = board.current_player
        self.chain_jump = board.chain_jump
        self.chain_jump_from = board.chain_jump_from

    def is_valid_position(self, row, column):
        if row >= 0 and row < CheckersBoard.board_height and column >= 0 and column < CheckersBoard.board_width:
            return True
        return False

    def is_empty_position(self, row, column):
        if self.is_valid_position(row, column):
            if self.p1_positions[row, column] == 0 and self.p2_positions[row, column] == 0:
                return True
        return False

    def get_player_positions(self, player):
        if player == 1:
            return self.p1_positions
        if player == -1:
            return self.p2_positions
        return None

    def get_players_kings(self, player):
        if player == 1:
            return self.p1_kings
        if player == -1:
            return self.p2_kings
        return None

    def get_players_pieces(self, player):
        if player == 1:
            return self.p1_pieces
        elif player == -1:
            return self.p2_pieces
        return None

    def is_valid_move(self, player, start, end):
        if not self.is_empty_position(end[0], end[1]) or not self.is_valid_position(start[0], start[1]):
            return False
        player_positions = self.get_player_positions(player)
        player_kings = self.get_players_kings(player)
        if player_positions[start[0], start[1]] == 0: #if no token at start
            return False
        if player_kings[start[0], start[1]] == 0:
            if start[0] - end[0] != player: #going in wrong direction
                return False
            if abs(start[1] - end[1]) != 1:
                return False
        else:
            if abs(start[0] - end[0]) != 1 or abs(start[1] - end[1]) != 1: #is not diagonally adjacent
                return False
        return True

    def execute_move(self, player, start, end):
        players_positions = self.get_player_positions(player)
        players_kings = self.get_players_kings(player)
        players_pieces = self.get_players_pieces(player)
        players_positions[start[0], start[1]] = 0
        players_positions[end[0], end[1]] = 1
        del players_pieces[(start[0], start[1])]
        players_pieces[(end[0], end[1])] = players_kings[start[0], start[1]]
        players_kings[end[0], end[1]] = players_kings[start[0], start[1]]
        players_kings[start[0], start[1]] = 0
        if (player == 1 and end[0] == 0) or (player == -1 and end[0] == 7):
            players_kings[end[0], end[1]] = 1
        self.moves_without_capture += 1
        self.p1_valid_moves_updated = False
        self.p2_valid_moves_updated = False
        self.current_player *= -1
        self.chain_jump = False
        self.chain_jump_from = None
        return True

    def get_valid_moves_from_pos(self, player, pos):
        valid_moves = []
        index_list = []
        row_offset = -player
        players_kings = self.get_players_kings(player)
        x = pos[0] + row_offset
        y = pos[1] - 1
        if self.is_valid_move(player, pos, [x, y]):
            new_move = CheckersBoard(board=self)
            new_move.execute_move(player, pos, [x, y])
            valid_moves.append(new_move)
            if row_offset == 1:
                index_list.append(self.move_to_index(pos, Direction.DOWN_LEFT.value))
            else:
                index_list.append(self.move_to_index(pos, Direction.UP_LEFT.value))
        y = pos[1] + 1
        if self.is_valid_move(player, pos, [x, y]):
            new_move = CheckersBoard(board=self)
            new_move.execute_move(player, pos, [x, y])
            valid_moves.append(new_move)
            if row_offset == 1:
                index_list.append(self.move_to_index(pos, Direction.DOWN_RIGHT.value))
            else:
                index_list.append(self.move_to_index(pos, Direction.UP_RIGHT.value))
        if players_kings[pos[0], pos[1]] == 1:
            x = pos[0] - row_offset
            y = pos[1] - 1
            if self.is_valid_move(player, pos, [x, y]):
                new_move = CheckersBoard(board=self)
                new_move.execute_move(player, pos, [x, y])
                valid_moves.append(new_move)
                if row_offset == 1:
                    index_list.append(self.move_to_index(pos, Direction.UP_LEFT.value))
                else:
                    index_list.append(self.move_to_index(pos, Direction.DOWN_LEFT.value))
            y = pos[1] + 1
            if self.is_valid_move(player, pos, [x, y]):
                new_move = CheckersBoard(board=self)
                new_move.execute_move(player, pos, [x, y])
                valid_moves.append(new_move)
                if row_offset == 1:
                    index_list.append(self.move_to_index(pos, Direction.UP_RIGHT.value))
                else:
                    index_list.append(self.move_to_index(pos, Direction.DOWN_RIGHT.value))
        return valid_moves, index_list

    def execute_jump(self, player, start, end):
        if not self.is_valid_jump(player, start, end):
            return False
        player_positions = self.get_player_positions(player)
        players_kings = self.get_players_kings(player)
        player_pieces = self.get_players_pieces(player)
        opponent_positions = self.get_player_positions(-player)
        opponent_kings = self.get_players_kings(-player)
        opponent_pieces = self.get_players_pieces(-player)
        x = int(start[0] + ((end[0] - start[0]) / 2))
        y = int(start[1] + ((end[1] - start[1]) / 2))
        player_positions[start[0], start[1]] = 0
        player_positions[end[0], end[1]] = 1
        del player_pieces[(start[0], start[1])]
        player_pieces[(end[0], end[1])] = players_kings[start[0], start[1]]
        players_kings[end[0], end[1]] = players_kings[start[0], start[1]]
        players_kings[start[0], start[1]] = 0
        opponent_positions[x, y] = 0
        opponent_kings[x, y] = 0
        del opponent_pieces[(x, y)]
        if self.can_jump(player, end):
            self.chain_jump = True
            self.chain_jump_from = end
        else:
            self.chain_jump = False
            self.current_player *= -1
        if (player == 1 and end[0] == 0) or (player == -1 and end[0] == 7):
            players_kings[end[0], end[1]] = 1
        self.moves_without_capture = 0
        self.p1_valid_moves_updated = False
        self.p2_valid_moves_updated = False
        return True

    def is_valid_jump(self, player, start, end):
        # if abs(start[0] - end[0]) != 2 or abs(start[1] - end[1]) != 2:
            # return False
        if not self.is_empty_position(end[0], end[1]):
            return False
        # player_positions = self.get_player_positions(player)
        opponent_positions = self.get_player_positions(-player)
        x1 = int(start[0] + ((end[0] - start[0]) / 2))
        y1 = int(start[1] + ((end[1] - start[1]) / 2))
        if opponent_positions[x1, y1] != 1:
            return False
        players_kings = self.get_players_kings(player)
        if players_kings[start[0], start[1]] == 0 and start[0] - x1 != player:
            return False
        return True

    def get_jumps(self, player, pos, include_chain_jumps=True):
        jumps = []
        chain_jumps = []
        index_list = []
        if self.is_valid_jump(player, pos, [pos[0] + 2, pos[1] + 2]):
            index_list.append(self.move_to_index(pos, Direction.DOWN_RIGHT.value, jump=True))
            new_jump = CheckersBoard(board=self)
            if new_jump.execute_jump(player, pos, [pos[0] + 2, pos[1] + 2]):
                if include_chain_jumps:
                    chain_jumps, _ = new_jump.get_jumps(player, [pos[0] + 2, pos[1] + 2])
                if len(chain_jumps) > 0 and include_chain_jumps:
                    jumps = np.hstack((jumps, chain_jumps))
                else:
                    jumps = np.append(jumps, new_jump)
        if self.is_valid_jump(player, pos, [pos[0] + 2, pos[1] - 2]):
            index_list.append(self.move_to_index(pos, Direction.DOWN_LEFT.value, jump=True))
            new_jump = CheckersBoard(board=self)
            if new_jump.execute_jump(player, pos, [pos[0] + 2, pos[1] - 2]):
                if include_chain_jumps:
                    chain_jumps, _ = new_jump.get_jumps(player, [pos[0] + 2, pos[1] - 2])
                if len(chain_jumps) > 0 and include_chain_jumps:
                    jumps = np.hstack((jumps, chain_jumps))
                else:
                    jumps = np.append(jumps, new_jump)
        if self.is_valid_jump(player, pos, [pos[0] - 2, pos[1] + 2]):
            index_list.append(self.move_to_index(pos, Direction.UP_RIGHT.value, jump=True))
            new_jump = CheckersBoard(board=self)
            if new_jump.execute_jump(player, pos, [pos[0] - 2, pos[1] + 2]):
                if include_chain_jumps:
                    chain_jumps, _ = new_jump.get_jumps(player, [pos[0] - 2, pos[1] + 2])
                if len(chain_jumps) > 0 and include_chain_jumps:
                    jumps = np.hstack((jumps, chain_jumps))
                else:
                    jumps = np.append(jumps, new_jump)
        if self.is_valid_jump(player, pos, [pos[0] - 2, pos[1] - 2]):
            index_list.append(self.move_to_index(pos, Direction.UP_LEFT.value, jump=True))
            new_jump = CheckersBoard(board=self)
            if new_jump.execute_jump(player, pos, [pos[0] - 2, pos[1] - 2]):
                if include_chain_jumps:
                    chain_jumps, _ = new_jump.get_jumps(player, [pos[0] - 2, pos[1] - 2])
                if len(chain_jumps) > 0 and include_chain_jumps:
                    jumps = np.hstack((jumps, chain_jumps))
                else:
                    jumps = np.append(jumps, new_jump)
        return jumps, index_list

    def get_valid_moves(self, player, include_index=False, include_chain_jumps=True):
        if player == 1 and self.p1_valid_moves_updated is False:
            self.update_valid_moves(player)
        elif player == -1 and self.p2_valid_moves_updated is False:
            self.update_valid_moves(player)
        if include_chain_jumps is False:
            if player == 1:
                if len(self.p1_valid_moves_single_jumps) > 0:
                    if include_index:
                        return self.p1_valid_moves_single_jumps
                    else:
                        return [self.p1_valid_moves_single_jumps[x][0] for x in range(len(self.p1_valid_moves_single_jumps))]
            if player == -1:
                if len(self.p2_valid_moves_single_jumps) > 0:
                    if include_index:
                        return self.p2_valid_moves_single_jumps
                    else:
                        return [self.p2_valid_moves_single_jumps[x][0] for x in range(len(self.p2_valid_moves_single_jumps))]
        if player == 1:
            if include_index:
                return self.p1_valid_moves
            else:
                moves = [self.p1_valid_moves[x][0] for x in range(len(self.p1_valid_moves))]
                return moves
        elif player == -1:
            if include_index:
                return self.p2_valid_moves
            else:
                moves = [self.p2_valid_moves[x][0] for x in range(len(self.p2_valid_moves))]
                return moves
        return None

    def update_valid_moves(self, player):
        valid_moves = []
        move_indices = []
        jump_indices = []
        jumps = []
        single_jumps = []
        player_pieces = self.get_players_pieces(player)
        if self.chain_jump:
            new_jumps, new_jump_indices = self.get_jumps(player, self.chain_jump_from)
            new_single_jumps, single_jump_indices = self.get_jumps(player, self.chain_jump_from, include_chain_jumps=False)
            if player == 1:
                self.p1_valid_moves = list(zip(new_jumps, new_jump_indices))
                self.p1_valid_moves_single_jumps = list(zip(new_single_jumps, single_jump_indices))
                self.p1_valid_moves_updated = True
                return
            elif player == -1:
                self.p2_valid_moves = list(zip(new_jumps, new_jump_indices))
                self.p2_valid_moves_single_jumps = list(zip(new_single_jumps, single_jump_indices))
                self.p2_valid_moves_updated = True
                return
        for (row, column), king in player_pieces.items():
            new_moves, new_move_indices = self.get_valid_moves_from_pos(player, [row, column])
            new_jumps, new_jump_indices = self.get_jumps(player, [row, column])
            new_single_jumps, single_jump_indices = self.get_jumps(player, [row, column], include_chain_jumps=False)
            if len(new_moves) > 0:
                valid_moves = np.hstack((valid_moves, new_moves))
                move_indices = np.hstack((move_indices, new_move_indices))
            if len(new_jumps) > 0:
                jumps = np.hstack((jumps, new_jumps))
                jump_indices = np.hstack((jump_indices, new_jump_indices))
            if len(new_single_jumps) > 0:
                single_jumps = np.hstack((single_jumps, new_single_jumps))
        if len(jumps) > 0:
            if player == 1:
                self.p1_valid_moves = list(zip(jumps, jump_indices))
                self.p1_valid_moves_single_jumps = list(zip(single_jumps, jump_indices))
                self.p1_valid_moves_updated = True
            elif player == -1:
                self.p2_valid_moves = list(zip(jumps, jump_indices))
                self.p2_valid_moves_single_jumps = list(zip(single_jumps, jump_indices))
                self.p2_valid_moves_updated = True
        else:
            if player == 1:
                self.p1_valid_moves = list(zip(valid_moves, move_indices))
                self.p1_valid_moves_updated = True
            elif player == -1:
                self.p2_valid_moves = list(zip(valid_moves, move_indices))
                self.p2_valid_moves_updated = True

    def can_jump(self, player, pos):
        if self.is_valid_jump(player, pos, [pos[0] + 2, pos[1] + 2]):
            return True
        if self.is_valid_jump(player, pos, [pos[0] + 2, pos[1] - 2]):
            return True
        if self.is_valid_jump(player, pos, [pos[0] - 2, pos[1] + 2]):
            return True
        if self.is_valid_jump(player, pos, [pos[0] - 2, pos[1] - 2]):
            return True
        return False

    def has_valid_moves_from_pos(self, player, pos):
        row_offset = -player
        x = pos[0] + row_offset
        y = pos[1] - 1
        if self.is_valid_move(player, pos, [x, y]):
            return True
        y = pos[1] + 1
        if self.is_valid_move(player, pos, [x, y]):
            return True
        players_kings = self.get_players_kings(player)
        if players_kings[pos[0], pos[1]] == 1:
            x = pos[0] - row_offset
            y = pos[1] - 1
            if self.is_valid_move(player, pos, [x, y]):
                return True
            y = pos[1] + 1
            if self.is_valid_move(player, pos, [x, y]):
                return True
        return False

    def has_valid_moves(self, player):
        player_pieces = self.get_players_pieces(player)
        for pos, king in player_pieces.items():
            if self.has_valid_moves_from_pos(player, [pos[0], pos[1]]):
                return True
            if self.can_jump(player, [pos[0], pos[1]]):
                return True
        return False

    def game_ended(self):
        p1_pieces = 0
        for p in self.p1_positions.flat:
            p1_pieces += p
        if p1_pieces == 0:
            return True, -1
        p2_pieces = 0
        for p in self.p2_positions.flat:
            p2_pieces += p
        if p2_pieces == 0:
            return True, 1
        if not self.has_valid_moves(1):
            return True, -1
        if not self.has_valid_moves(-1):
            return True, 1
        if self.moves_without_capture >= 50:
            return True, 0
        return False, None

    @staticmethod
    def move_to_index(start_pos, direction, jump=False):
        tile_index = (start_pos[0] * 4) + (int(start_pos[1] / 2))
        index = (tile_index * 4) + direction
        if jump:
            index += 128
        return int(index)
