import numpy as np
from numba import njit
from board import (
    get_all_legal_moves_njit, 
    move_piece_njit, 
    king_check_check_njit
)
from evaluate import evaluate_board

@njit
def move_order_score_njit(state, move):
    # move is (from_row, from_col, to_row, to_col, promotion)
    target_piece = state[move[2], move[3]]
    score = 0
    # PROMOTIONS FIRST
    if move[4] != 0:
        score += 100
    # CAPTURES FIRST
    if target_piece != 0 and target_piece != 7 and target_piece != -7:
        score += abs(target_piece) * 10
    return score

@njit
def minmax_njit(state, team, depth, alpha, beta, 
                wkm, bkm, wram, wrhm, bram, brhm):
    
    legal_moves = get_all_legal_moves_njit(state, team, wkm, bkm, wram, wrhm, bram, brhm)

    if len(legal_moves) == 0:
        wic, bic, _, _ = king_check_check_njit(state)
        if team == 1:
            if wic:
                return -1000.0 - depth, (0,0,0,0,0)
            else:
                return 0.0, (0,0,0,0,0)
        if team == -1:
            if bic:
                return 1000.0 + depth, (0,0,0,0,0)
            else:
                return 0.0, (0,0,0,0,0)

    if depth == 0:
        return evaluate_board(state), (0,0,0,0,0)

    # Sort moves (simplified sort for numba)
    scores = np.zeros(len(legal_moves))
    for i in range(len(legal_moves)):
        scores[i] = move_order_score_njit(state, legal_moves[i])
    
    # Sort indices by score descending
    indices = np.argsort(scores)[::-1]
    
    best_eval = -99999.0 if team == 1 else 99999.0
    best_move = legal_moves[indices[0]]

    for idx in indices:
        m = legal_moves[idx]
        ns, nwkm, nbkm, nwram, nwrhm, nbram, nbrhm = move_piece_njit(
            state, m[0], m[1], m[2], m[3], m[4],
            wkm, bkm, wram, wrhm, bram, brhm
        )

        ev, _ = minmax_njit(ns, team * -1, depth - 1, alpha, beta,
                            nwkm, nbkm, nwram, nwrhm, nbram, nbrhm)

        if team == 1:
            if ev > best_eval:
                best_eval = ev
                best_move = m
            if best_eval > alpha:
                alpha = best_eval
        else:
            if ev < best_eval:
                best_eval = ev
                best_move = m
            if best_eval < beta:
                beta = best_eval

        if beta <= alpha:
            break
    
    return best_eval, best_move

class Search:
    def __init__(self):
        pass

    def minmax(self, board, team, depth, alpha=-99999, beta=99999):
        ev, move = minmax_njit(
            board.state, team, depth, alpha, beta,
            board.white_king_moved, board.black_king_moved,
            board.white_rook_a_moved, board.white_rook_h_moved,
            board.black_rook_a_moved, board.black_rook_h_moved
        )
        
        # Check if dummy move was returned
        if move == (0, 0, 0, 0, 0):
            return ev, None

        # Convert move back to None for promotion if it was 0
        if move[4] == 0:
            move = (move[0], move[1], move[2], move[3], None)
        return ev, move
