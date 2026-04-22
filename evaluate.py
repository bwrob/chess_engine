import numpy as np
from numba import njit

@njit
def evaluate_board(state):
    score = 0.0

    for ri in range(8):
        for ci in range(8):
            p = state[ri, ci]

            if p == 0:
                continue

            team = 1 if p > 0 else -1

            if p == 1 or p == -1: ## PAWNS
                score += 1.0 * team

                ## DOUBLED PAWNS
                if team == 1:
                    if ri + 1 <= 7:
                        if state[ri + 1, ci] == 1:
                            score -= 0.5
                elif team == -1:
                    if ri - 1 >= 0:
                        if state[ri - 1, ci] == -1:
                            score += 0.5

            elif p == 2 or p == -2: ## KNIGHTS
                score += 3.0 * team

                if (ri == 3 or ri == 4) and (ci == 3 or ci == 4): ## CENTER
                    score += 0.5 * team
                if (ri == 2 or ri == 5) and (ci == 2 or ci == 5): ## SEMI-CENTER
                    score += 0.25 * team
                if (ri == 1 or ri == 6) and (ci == 1 or ci == 6): ## OUTER
                    score -= 0.25 * team
                if (ri == 0 or ri == 7) and (ci == 0 or ci == 7): ## CORNERS
                    score -= 0.5 * team

            elif p == 3 or p == -3: ## BISHOPS
                score += 3.0 * team

                if (ri == 3 or ri == 4) and (ci == 3 or ci == 4): ## CENTER
                    score += 0.5 * team
                if (ri == 2 or ri == 5) and (ci == 2 or ci == 5): ## SEMI-CENTER
                    score += 0.25 * team

            elif p == 4 or p == -4: ## ROOKS
                score += 5.0 * team

            elif p == 5 or p == -5: ## QUEENS
                score += 9.0 * team

            elif p == 6 or p == -6: ## KINGS
                if team == 1:
                    if (ri == 7 and ci == 2) or (ri == 7 and ci == 6): ## SAFE
                        score += 0.75
                    if (ri == 7 and ci == 1) or (ri == 7 and ci == 7): ## SEMI-SAFE
                        score += 0.25

                if team == -1:
                    if (ri == 0 and ci == 2) or (ri == 0 and ci == 6): ## SAFE
                        score -= 0.75
                    if (ri == 0 and ci == 1) or (ri == 0 and ci == 7): ## SEMI-SAFE
                        score -= 0.25

    return score

class Evaluate:
    def evaluate(self, board):
        return evaluate_board(board.state)