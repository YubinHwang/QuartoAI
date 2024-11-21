import numpy as np
import random
from itertools import product

import time

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
    
    # 상대한테 가장 불리한 말 선택
    def select_piece(self):
        selected_piece = None
        best_score = 1e9    # 최소화시켜서 상대에게 불리하게 함
        for piece in self.available_pieces:
            copy_board = np.copy(self.board)    # 원본 유지하면서 서치
            # minimax 알고리즘 -> 각 말 점수 계산
            score = self.minmax_ab(copy_board, 2, False, piece, -1e9, 1e9)
            if score < best_score:
                best_score = score
                selected_piece = piece
        return selected_piece

    # 선택된 말을 놓을 최적 위치 선정
    def place_piece(self, selected_piece):
        best_move = None
        best_score = -1e9   # 최대화시켜서 자신에게 유리하게 함
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0:   #빈 칸인 경우
                    self.board[row][col] = self.pieces.index(selected_piece) + 1 # 선택한 말이 전체 말 중에 몇 번째에 있는 지 찾고 위치지정
                    # minimax 알고리즘 -> 각 말 위치 점수 계산
                    score = self.minmax_ab(self.board, 2, False, selected_piece, -1e9, 1e9)
                    self.board[row][col] = 0    # 원래대로 지워줌 
                    if score > best_score:
                        best_score = score
                        best_move = (row, col)
        return best_move  
    
    
    # 승리 조건 확인 
    def is_board_full(self):
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0:
                    return False
        return True

    def check_line(self,line):
        if 0 in line:
            return False  # Incomplete line
        characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])
        for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
            if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
                return True
        return False

    def check_2x2_subgrid_win(self):
        for r in range(3):
            for c in range(3):
                subgrid = [self.board[r][c], self.board[r][c+1], self.board[r+1][c], self.board[r+1][c+1]]
                if 0 not in subgrid:  # All cells must be filled
                    characteristics = [self.pieces[idx - 1] for idx in subgrid]
                    for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
                        if len(set(char[i] for char in characteristics)) == 1:  # All share the same characteristic
                            return True
        return False

    def check_win(self):
        # Check rows, columns, and diagonals
        for col in range(4):
            if self.check_line([self.board[row][col] for row in range(4)]):
                return True
        
        for row in range(4):
            if self.check_line([self.board[row][col] for col in range(4)]):
                return True
            
        if self.check_line([self.board[i][i] for i in range(4)]) or self.check_line([self.board[i][3 - i] for i in range(3)]):
            return True

        # Check 2x2 sub-grids
        if self.check_2x2_subgrid_win():
            return True
        
        return False
        

    # minmax 알고리즘
    def minmax_ab(self, board, depth, is_maximizing, selected_piece, alpha, beta):

        # 종료 조건
        if self.check_win():
            if is_maximizing: #내 턴일때
                return 1  # 승리
            else:
                return -1  # 패배
        
        if self.is_board_full() or depth == 0:
            return 0  # 무승부

        #최대화 시키는 플레이어의 턴
        if is_maximizing:
            best_score = -1e9
            for row in range(4):
                for col in range(4):
                    if board[row][col] == 0:    # 빈 칸일 때
                        board[row][col] = self.pieces.index(selected_piece) + 1 #말 둬 보고
                        score = self.minmax_ab(board, depth - 1, False, selected_piece, alpha, beta)    # 상대의 점수를 계산
                        board[row][col] = 0 # 보드 원래대로
                        best_score = max(best_score, score)
                        alpha = max(alpha, best_score)  #알파값 갱신
                        if beta <= alpha:   # 가지치기
                            break
            return best_score
        
        #최소화 시키는 플레이어의 턴
        else:
            best_score = 1e9
            for row in range(4):
                for col in range(4):
                    if board[row][col] == 0:
                        board[row][col] = self.pieces.index(selected_piece) + 1
                        score = self.minmax_ab(board, depth - 1, True, selected_piece, alpha, beta)
                        board[row][col] = 0
                        best_score = min(best_score, score)
                        beta = min(beta, best_score)
                        if beta <= alpha:   # 가지치기
                            break
            return best_score