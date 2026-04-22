import numpy as np
from numba import njit

# PIECES
# 0 = empty
# 1 - white pawn
# 2 - white knight
# 3 - white bishop
# 4 - white rook
# 5 - white queen
# 6 - white king
# -1 - black pawn
# -2 - black knight
# -3 - black bishop
# -4 - black rook
# -5 - black queen
# -6 - black king

@njit
def king_check_check_njit(state):
    white_in_check = False
    black_in_check = False
    white_king_pos = (-1, -1)
    black_king_pos = (-1, -1)

    for row in range(8):
        for col in range(8):
            piece = state[row, col]
            if piece == 6:
                white_king_pos = (row, col)
            elif piece == -6:
                black_king_pos = (row, col)

    # White king check
    if white_king_pos != (-1, -1):
        row, col = white_king_pos
        # Pawns
        if row - 1 >= 0:
            if col + 1 <= 7 and state[row - 1, col + 1] == -1: white_in_check = True
            if col - 1 >= 0 and state[row - 1, col - 1] == -1: white_in_check = True
        # Knights
        knight_moves = ((row + 2, col + 1), (row + 2, col - 1), (row - 2, col + 1), (row - 2, col - 1), 
                        (row + 1, col + 2), (row + 1, col - 2), (row - 1, col + 2), (row - 1, col - 2))
        for r, c in knight_moves:
            if 0 <= r <= 7 and 0 <= c <= 7 and state[r, c] == -2: white_in_check = True
        # Long range
        dirs = ((-1, -1), (-1, 1), (1, -1), (1, 1), (1, 0), (-1, 0), (0, 1), (0, -1))
        for dx, dy in dirs:
            cr, cc = row + dx, col + dy
            while 0 <= cr <= 7 and 0 <= cc <= 7:
                cp = state[cr, cc]
                if cp != 0:
                    if (dx != 0 and dy != 0): # diagonal
                        if cp == -3 or cp == -5: white_in_check = True
                    else: # straight
                        if cp == -4 or cp == -5: white_in_check = True
                    break
                cr += dx
                cc += dy
        # King
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0: continue
                r, c = row + dr, col + dc
                if 0 <= r <= 7 and 0 <= c <= 7 and state[r, c] == -6: white_in_check = True

    # Black king check
    if black_king_pos != (-1, -1):
        row, col = black_king_pos
        # Pawns
        if row + 1 <= 7:
            if col + 1 <= 7 and state[row + 1, col + 1] == 1: black_in_check = True
            if col - 1 >= 0 and state[row + 1, col - 1] == 1: black_in_check = True
        # Knights
        knight_moves = ((row + 2, col + 1), (row + 2, col - 1), (row - 2, col + 1), (row - 2, col - 1), 
                        (row + 1, col + 2), (row + 1, col - 2), (row - 1, col + 2), (row - 1, col - 2))
        for r, c in knight_moves:
            if 0 <= r <= 7 and 0 <= c <= 7 and state[r, c] == 2: black_in_check = True
        # Long range
        dirs = ((-1, -1), (-1, 1), (1, -1), (1, 1), (1, 0), (-1, 0), (0, 1), (0, -1))
        for dx, dy in dirs:
            cr, cc = row + dx, col + dy
            while 0 <= cr <= 7 and 0 <= cc <= 7:
                cp = state[cr, cc]
                if cp != 0:
                    if (dx != 0 and dy != 0): # diagonal
                        if cp == 3 or cp == 5: black_in_check = True
                    else: # straight
                        if cp == 4 or cp == 5: black_in_check = True
                    break
                cr += dx
                cc += dy
        # King
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0: continue
                r, c = row + dr, col + dc
                if 0 <= r <= 7 and 0 <= c <= 7 and state[r, c] == 6: black_in_check = True

    return white_in_check, black_in_check, white_king_pos, black_king_pos

@njit
def move_piece_njit(state, row, col, new_row, new_col, promotion_piece, 
                    white_king_moved, black_king_moved, 
                    white_rook_a_moved, white_rook_h_moved, 
                    black_rook_a_moved, black_rook_h_moved):
    
    piece = state[row, col]
    team = 1 if piece > 0 else -1
    target_piece = state[new_row, new_col]

    new_state = state.copy()
    
    # CASTLING
    if piece == 6 and row == 7 and col == 4 and new_row == 7 and new_col == 6:
        new_state[7, 4] = 0
        new_state[7, 6] = 6
        new_state[7, 5] = 4
        new_state[7, 7] = 0
    elif piece == 6 and row == 7 and col == 4 and new_row == 7 and new_col == 2:
        new_state[7, 4] = 0
        new_state[7, 2] = 6
        new_state[7, 3] = 4
        new_state[7, 0] = 0
    elif piece == -6 and row == 0 and col == 4 and new_row == 0 and new_col == 6:
        new_state[0, 4] = 0
        new_state[0, 6] = -6
        new_state[0, 5] = -4
        new_state[0, 7] = 0
    elif piece == -6 and row == 0 and col == 4 and new_row == 0 and new_col == 2:
        new_state[0, 4] = 0
        new_state[0, 2] = -6
        new_state[0, 3] = -4
        new_state[0, 0] = 0
    # EN PASSANT CAPTURE
    elif piece == 1 and target_piece == -7:
        new_state[row, col] = 0
        new_state[new_row, new_col] = 1 if promotion_piece == 0 else promotion_piece
        new_state[new_row + 1, new_col] = 0
    elif piece == -1 and target_piece == 7:
        new_state[row, col] = 0
        new_state[new_row, new_col] = -1 if promotion_piece == 0 else -promotion_piece
        new_state[new_row - 1, new_col] = 0
    else:
        new_state[row, col] = 0
        if promotion_piece != 0:
            new_state[new_row, new_col] = promotion_piece if team == 1 else -promotion_piece
        else:
            new_state[new_row, new_col] = piece

    # CLEAR OLD EN PASSANT MARKERS
    for i in range(8):
        for j in range(8):
            if new_state[i, j] == 7 or new_state[i, j] == -7:
                new_state[i, j] = 0

    # NEW EN PASSANT MARKERS
    if piece == 1 and new_row == row - 2:
        new_state[row - 1, col] = 7
    elif piece == -1 and new_row == row + 2:
        new_state[row + 1, col] = -7

    # UPDATE FLAGS
    if piece == 6: white_king_moved = True
    elif piece == -6: black_king_moved = True
    elif piece == 4 and row == 7 and col == 0: white_rook_a_moved = True
    elif piece == 4 and row == 7 and col == 7: white_rook_h_moved = True
    elif piece == -4 and row == 0 and col == 0: black_rook_a_moved = True
    elif piece == -4 and row == 0 and col == 7: black_rook_h_moved = True

    return new_state, white_king_moved, black_king_moved, white_rook_a_moved, white_rook_h_moved, black_rook_a_moved, black_rook_h_moved

@njit
def get_legal_moves_njit(state, row, col, white_king_moved, black_king_moved, 
                         white_rook_a_moved, white_rook_h_moved, 
                         black_rook_a_moved, black_rook_h_moved):
    piece = state[row, col]
    team = 1 if piece > 0 else -1
    moves = []

    if piece == 1: # White Pawn
        if row - 1 >= 0:
            if state[row - 1, col] == 0:
                if row - 1 == 0:
                    for p in [2, 3, 4, 5]: moves.append((row - 1, col, p))
                else:
                    moves.append((row - 1, col, 0))
                    if row == 6 and state[row - 2, col] == 0:
                        moves.append((row - 2, col, 0))
            for dc in [-1, 1]:
                if 0 <= col + dc <= 7:
                    tp = state[row - 1, col + dc]
                    if tp < 0:
                        if row - 1 == 0:
                            for p in [2, 3, 4, 5]: moves.append((row - 1, col + dc, p))
                        else:
                            moves.append((row - 1, col + dc, 0))
    elif piece == -1: # Black Pawn
        if row + 1 <= 7:
            if state[row + 1, col] == 0:
                if row + 1 == 7:
                    for p in [2, 3, 4, 5]: moves.append((row + 1, col, p))
                else:
                    moves.append((row + 1, col, 0))
                    if row == 1 and state[row + 2, col] == 0:
                        moves.append((row + 2, col, 0))
            for dc in [-1, 1]:
                if 0 <= col + dc <= 7:
                    tp = state[row + 1, col + dc]
                    if tp > 0:
                        if row + 1 == 7:
                            for p in [2, 3, 4, 5]: moves.append((row + 1, col + dc, p))
                        else:
                            moves.append((row + 1, col + dc, 0))
    elif abs(piece) == 2: # Knight
        for dr, dc in [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr <= 7 and 0 <= nc <= 7:
                if (team == 1 and state[nr, nc] <= 0) or (team == -1 and state[nr, nc] >= 0):
                    moves.append((nr, nc, 0))
    elif abs(piece) == 3: # Bishop
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            cr, cc = row + dr, col + dc
            while 0 <= cr <= 7 and 0 <= cc <= 7:
                cp = state[cr, cc]
                if cp == 0: moves.append((cr, cc, 0))
                else:
                    if (team > 0 and cp < 0) or (team < 0 and cp > 0): moves.append((cr, cc, 0))
                    break
                cr += dr
                cc += dc
    elif abs(piece) == 4: # Rook
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            cr, cc = row + dr, col + dc
            while 0 <= cr <= 7 and 0 <= cc <= 7:
                cp = state[cr, cc]
                if cp == 0: moves.append((cr, cc, 0))
                else:
                    if (team > 0 and cp < 0) or (team < 0 and cp > 0): moves.append((cr, cc, 0))
                    break
                cr += dr
                cc += dc
    elif abs(piece) == 5: # Queen
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1),(1,0),(-1,0),(0,1),(0,-1)]:
            cr, cc = row + dr, col + dc
            while 0 <= cr <= 7 and 0 <= cc <= 7:
                cp = state[cr, cc]
                if cp == 0: moves.append((cr, cc, 0))
                else:
                    if (team > 0 and cp < 0) or (team < 0 and cp > 0): moves.append((cr, cc, 0))
                    break
                cr += dr
                cc += dc
    elif abs(piece) == 6: # King
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0: continue
                nr, nc = row + dr, col + dc
                if 0 <= nr <= 7 and 0 <= nc <= 7:
                    if (team == 1 and state[nr, nc] <= 0) or (team == -1 and state[nr, nc] >= 0):
                        moves.append((nr, nc, 0))
        # Castling
        if team == 1 and not white_king_moved:
            if not white_rook_h_moved and state[7, 7] == 4 and state[7, 5] == 0 and state[7, 6] == 0:
                moves.append((7, 6, 0))
            if not white_rook_a_moved and state[7, 0] == 4 and state[7, 1] == 0 and state[7, 2] == 0 and state[7, 3] == 0:
                moves.append((7, 2, 0))
        elif team == -1 and not black_king_moved:
            if not black_rook_h_moved and state[0, 7] == -4 and state[0, 5] == 0 and state[0, 6] == 0:
                moves.append((0, 6, 0))
            if not black_rook_a_moved and state[0, 0] == -4 and state[0, 1] == 0 and state[0, 2] == 0 and state[0, 3] == 0:
                moves.append((0, 2, 0))

    legal_moves = []
    for m in moves:
        nr, nc, p = m
        ns, wkm, bkm, wram, wrhm, bram, brhm = move_piece_njit(state, row, col, nr, nc, p, 
                                                               white_king_moved, black_king_moved, 
                                                               white_rook_a_moved, white_rook_h_moved, 
                                                               black_rook_a_moved, black_rook_h_moved)
        wic, bic, _, _ = king_check_check_njit(ns)
        if team == 1 and not wic:
            if piece == 6 and row == 7 and col == 4:
                if nr == 7 and nc == 6:
                    wic_orig, _, _, _ = king_check_check_njit(state)
                    ns_mid, _, _, _, _, _, _ = move_piece_njit(state, 7, 4, 7, 5, 0, wkm, bkm, wram, wrhm, bram, brhm)
                    wic_mid, _, _, _ = king_check_check_njit(ns_mid)
                    if not wic_orig and not wic_mid: legal_moves.append(m)
                elif nr == 7 and nc == 2:
                    wic_orig, _, _, _ = king_check_check_njit(state)
                    ns_mid, _, _, _, _, _, _ = move_piece_njit(state, 7, 4, 7, 3, 0, wkm, bkm, wram, wrhm, bram, brhm)
                    wic_mid, _, _, _ = king_check_check_njit(ns_mid)
                    if not wic_orig and not wic_mid: legal_moves.append(m)
                else: legal_moves.append(m)
            else: legal_moves.append(m)
        elif team == -1 and not bic:
            if piece == -6 and row == 0 and col == 4:
                if nr == 0 and nc == 6:
                    _, bic_orig, _, _ = king_check_check_njit(state)
                    ns_mid, _, _, _, _, _, _ = move_piece_njit(state, 0, 4, 0, 5, 0, wkm, bkm, wram, wrhm, bram, brhm)
                    _, bic_mid, _, _ = king_check_check_njit(ns_mid)
                    if not bic_orig and not bic_mid: legal_moves.append(m)
                elif nr == 0 and nc == 2:
                    _, bic_orig, _, _ = king_check_check_njit(state)
                    ns_mid, _, _, _, _, _, _ = move_piece_njit(state, 0, 4, 0, 3, 0, wkm, bkm, wram, wrhm, bram, brhm)
                    _, bic_mid, _, _ = king_check_check_njit(ns_mid)
                    if not bic_orig and not bic_mid: legal_moves.append(m)
                else: legal_moves.append(m)
            else: legal_moves.append(m)

    return legal_moves

@njit
def get_all_legal_moves_njit(state, team, white_king_moved, black_king_moved, 
                             white_rook_a_moved, white_rook_h_moved, 
                             black_rook_a_moved, black_rook_h_moved):
    all_moves = []
    for r in range(8):
        for c in range(8):
            if state[r, c] * team > 0:
                piece_moves = get_legal_moves_njit(state, r, c, white_king_moved, black_king_moved, 
                                                   white_rook_a_moved, white_rook_h_moved, 
                                                   black_rook_a_moved, black_rook_h_moved)
                for m in piece_moves:
                    all_moves.append((r, c, m[0], m[1], m[2]))
    return all_moves

class Board:
    def __init__(self):
        self.state = np.array([[-4,-2,-3,-5,-6,-3,-2,-4],
                               [-1,-1,-1,-1,-1,-1,-1,-1],
                               [ 0, 0, 0, 0, 0, 0, 0, 0],
                               [ 0, 0, 0, 0, 0, 0, 0, 0],
                               [ 0, 0, 0, 0, 0, 0, 0, 0],
                               [ 0, 0, 0, 0, 0, 0, 0, 0],
                               [ 1, 1, 1, 1, 1, 1, 1, 1],
                               [ 4, 2, 3, 5, 6, 3, 2, 4]], dtype=np.int8)
        
        self.white_king_moved = False
        self.black_king_moved = False
        self.white_rook_a_moved = False
        self.white_rook_h_moved = False
        self.black_rook_a_moved = False
        self.black_rook_h_moved = False
                
    def move_piece(self, row, column, new_row, new_column, promotion_piece=None):
        if promotion_piece is None: promotion_piece = 0
        self.state, self.white_king_moved, self.black_king_moved, \
        self.white_rook_a_moved, self.white_rook_h_moved, \
        self.black_rook_a_moved, self.black_rook_h_moved = move_piece_njit(
            self.state, row, column, new_row, new_column, promotion_piece,
            self.white_king_moved, self.black_king_moved,
            self.white_rook_a_moved, self.white_rook_h_moved,
            self.black_rook_a_moved, self.black_rook_h_moved
        )

    def king_check_check(self):
        wic, bic, wkp, bkp = king_check_check_njit(self.state)
        # Convert -1, -1 back to None
        wkp = wkp if wic else None
        bkp = bkp if bic else None
        return wic, bic, wkp, bkp

    def get_legal_moves(self, row, column):
        moves = get_legal_moves_njit(self.state, row, column, 
                                     self.white_king_moved, self.black_king_moved, 
                                     self.white_rook_a_moved, self.white_rook_h_moved, 
                                     self.black_rook_a_moved, self.black_rook_h_moved)
        # Convert 0 back to None for promotion piece
        return [(m[0], m[1], m[2] if m[2] != 0 else None) for m in moves]

    def get_all_legal_moves(self, team):
        moves = get_all_legal_moves_njit(self.state, team, 
                                         self.white_king_moved, self.black_king_moved, 
                                         self.white_rook_a_moved, self.white_rook_h_moved, 
                                         self.black_rook_a_moved, self.black_rook_h_moved)
        return [(m[0], m[1], m[2], m[3], m[4] if m[4] != 0 else None) for m in moves]
    
    def check_for_win(self, team):
        all_legal_moves = self.get_all_legal_moves_njit_wrapper(team)
        if len(all_legal_moves) == 0:
            wic, bic, _, _ = king_check_check_njit(self.state)
            if team == 1:
                return -1 if wic else 0
            else:
                return 1 if bic else 0
        return None

    def get_all_legal_moves_njit_wrapper(self, team):
        return get_all_legal_moves_njit(self.state, team, 
                                        self.white_king_moved, self.black_king_moved, 
                                        self.white_rook_a_moved, self.white_rook_h_moved, 
                                        self.black_rook_a_moved, self.black_rook_h_moved)
