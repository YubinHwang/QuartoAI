import numpy as np
import random
from itertools import product

import time

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
    
    def select_piece(self):
        # Make your own algorithm here

        if(self.board.max()):
            print("여기에 로직 계속 작성")
            return random.choice(self.available_pieces)
        # 첫 선택이면 그냥 무조건 ENFJ를 준다    
        else:
            return (1,0,1,1) #ENFJ
            

    def place_piece(self, selected_piece):
        # selected_piece: The selected piece that you have to place on the board (e.g. (1, 0, 1, 0)).
        
        # Available locations to place the piece
        # 첫 수가 아닌 경우
        if(self.board.max()):
            row = self.evaluate_position(selected_piece) // 4
            col = self.evaluate_position(selected_piece) % 4
            return(row, col)

        else: # 내가 첫수라면 무조건 1,1에 둔다
            return (1,1)

    # 완성도 체크 함수
    def check_line_possibility(self, line):
        line = [grid for grid in line if grid != 0]
        characteristics = [self.pieces[idx - 1] for idx in line]
        for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
            if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
                return(len(line)) # 현재 line에서 맞는 쌍이 있는지, 몇개인지 확인
            else:
                return(-1) # -1인 곳은 확인할 필요 없음(전부 안맞는 곳임)


    def check_possibilities(self):
        # 이 함수에서 2이상이 나온 부분을 우선 봐서 
        # 내가 선택하는 순서면 거기랑 안맞는걸 주고 내가 놓는 순서면
        # 3인 곳을 찾아서 놓는다
        self.row_eval = []
        self.col_evel = []
        self.cross_eval = [] # 좌->우 대각선, 우->좌 대각선
        self.subgrid_eval = [] # 2X2 grid
        
        for col in range(4):
            self.col_evel.append(self.check_line_possibility([self.board[row][col] for row in range(4)]))
    
        for row in range(4):
            self.row_eval.append(self.check_line_possibility([self.board[row][col] for col in range(4)]))

        self.cross_eval.append(self.check_line_possibility([self.board[i][i] for i in range(4)]))
        self.cross_eval.append(self.check_line_possibility([self.board[i][3 - i] for i in range(4)]))

        # Check 2x2 sub-grids
        for r in range(3):
            for c in range(3):
                subgrid = [self.board[r][c], self.board[r][c+1], self.board[r+1][c], self.board[r+1][c+1]]
                subgrid = [grid for grid in subgrid if grid != 0]
                characteristics = [self.pieces[idx - 1] for idx in subgrid]
                for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
                    if len(set(char[i] for char in characteristics)) == 1:  # All share the same characteristic
                        self.subgrid_eval.append(len(subgrid))
                    else: self.subgrid_eval.append(-1) # -1인 곳은 확인할 필요 없음


    # 위험한 곳의 위치를 파악하고 내가 갖고 있는 piece가 어디 들어가면 좋은지 확인
    def evaluate_position(self, selected_piece):
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        max_row = max(self.row_eval) # 0,1,2,3
        max_col = max(self.col_evel) # 0,1,2,3
        max_cross = max(self.cross_eval) # 0, 1
        max_subgrid = max(self.subgrid_eval) # 0~8까지
        max_vals = [max_row, max_col, max_cross, max_subgrid]
        row_most_pos = self.row_eval.index(max_row)
        col_most_pos = self.col_evel.index(max_col)
        cross_most_pos = self.cross_eval.index(max_cross)
        subgrid_most_pos = self.subgrid_eval.index(max_subgrid)

        self.eval_board = np.zeros((4, 4), dtype=int)
        # **그냥 0이 아닌 부분에 뒀을 때 나한테 좋은지 안좋은지 eval_board에 업데이트

        # eval 값이 1이면 협력해도 됨, eval 값이 2이면 방해해야됨
        # 나에게 도움이 되는 칸은 +1, 도움이 안되는 칸은 -1한다
        # 마지막에 최댓값을 가지고 있는 칸을 return하면 됨
        
       
        if(max(max_vals) == 3):
            for loc in available_locs:
                r, c = loc
                pos_list = []
                self.board[r][c] = self.pieces.index(selected_piece) + 1
                pos_list.append(self.check_line_possibility([self.board[r][col] for col in range(4)]))
                pos_list.append(self.check_line_possibility([self.board[row][c] for row in range(4)]))
                pos_list.append(self.check_line_possibility([self.board[i][i] for i in range(4)]))
                pos_list.append(self.check_line_possibility([self.board[i][3-i] for i in range(4)]))
                if max(pos_list) == 4:
                    return (4*r + c)
            
        # row,col또는 row,cross 또는 col,cross에서 모두 eval 변수가 2인 순간 거의 게임 끝남
        # 이건 max값이 2인 경우니까 내가 두고 다음 차례에 4가 되지 않게 하는 말이 남았으면
        # 그 위치의 eval_board에 1을 더하고, 아니면 다음걸 평가한다.    
        if(max_row == 2):
            for i in range(4):
                if(self.board[row_most_pos][i] == 0):
                    self.board[row_most_pos][i] = self.pieces.index(selected_piece)
                    if(self.check_line_possibility([self.board[row_most_pos][col] for col in range(4)]) == 3):
                        # 4가 되지 않게 하는 말이 남아있어야됨
                        for idx, piece in enumerate(self.available_pieces):
                            self.board[row_most_pos][np.argmin(self.board[row_most_pos])] = self.pieces.index(piece) + 1
                            if (self.check_line_possibility([self.board[row_most_pos][col] for col in range(4)]) == 4):
                                self.eval_board[row_most_pos][i] -= 1
                            else: self.eval_board[row_most_pos][i] += 1
                            self.board[row_most_pos][np.argmin(self.board[row_most_pos])] = 0
                    self.board[row_most_pos][i] = 0

        if(max_col == 2):
            for i in range(4):
                if(self.board[i][col_most_pos] == 0):
                    self.board[i][col_most_pos] = self.pieces.index(selected_piece)
                    if(self.check_line_possibility([self.board[row][col_most_pos] for row in range(4)]) == 3):
                        # 4가 되지 않게 하는 말이 남아있어야됨
                        for idx, piece in enumerate(self.available_pieces):
                            self.board[np.argmin(self.board[col_most_pos])][col_most_pos] = self.pieces.index(piece) + 1
                            if (self.check_line_possibility([self.board[row][col_most_pos] for row in range(4)]) == 4):
                                self.eval_board[i][col_most_pos] -= 1
                            else: self.eval_board[i][col_most_pos] += 1
                            self.board[row_most_pos][np.argmin(self.board[row_most_pos])] = 0
                    self.board[i][max_col] = 0

        # 여기에 max_cross, max_subgird가 2일 때 로직을 작성
        if(max_cross == 2):
            for diag in range(2):  # diag = 0: 좌->우 대각선, diag = 1: 우->좌 대각선
                diagonal = ([self.board[i][i] for i in range(4)] if diag == 0
                            else [self.board[i][3 - i] for i in range(4)])
                for i in range(4):
                    if diagonal[i] == 0:  # 빈 칸을 찾음
                        r, c = (i, i) if diag == 0 else (i, 3 - i)
                        self.board[r][c] = self.pieces.index(selected_piece) + 1
                        if self.check_line_possibility([self.board[d][d] if diag == 0 else self.board[d][3 - d] for d in range(4)]) == 3:
                            # 4가 되지 않게 하는 말을 찾음
                            for idx, piece in enumerate(self.available_pieces):
                                self.board[r][c] = self.pieces.index(piece) + 1
                                if self.check_line_possibility([self.board[d][d] if diag == 0 else self.board[d][3 - d] for d in range(4)]) == 4:
                                    self.eval_board[r][c] -= 1
                                else:
                                    self.eval_board[r][c] += 1
                                self.board[r][c] = 0  # 복구
                        self.board[r][c] = 0  # 복구

        if(max_subgrid == 2):
            for r in range(3):
                for c in range(3):
                    subgrid = [self.board[r][c], self.board[r][c+1],
                            self.board[r+1][c], self.board[r+1][c+1]]
                    subgrid_indices = [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]
                    if subgrid.count(0) == 2:  # 빈 칸 2개인 서브그리드만 체크
                        for idx, (r, c) in enumerate(subgrid_indices):
                            if self.board[r][c] == 0:  # 빈 칸을 찾음
                                self.board[r][c] = self.pieces.index(selected_piece) + 1
                                # 해당 서브그리드가 승리 조건에 가까운지 평가
                                if self.check_line_possibility([self.board[i][j] for i, j in subgrid_indices]) == 3:
                                    # 4가 되지 않게 하는 말을 찾음
                                    for piece in self.available_pieces:
                                        self.board[r][c] = self.pieces.index(piece) + 1
                                        if self.check_line_possibility([self.board[i][j] for i, j in subgrid_indices]) == 4:
                                            self.eval_board[r][c] -= 1
                                        else:
                                            self.eval_board[r][c] += 1
                                        self.board[r][c] = 0  # 복구
                                self.board[r][c] = 0  # 복구

    
        # 내가 상대한테 말을 줘야되면 최악의 말
        
        # selected_piece를 다 놔보는데, 내가 두고 나서 각 eval 변수가 3이 되는 경우
        # 상대한테 줄 최악의 수가 남아 있어야한다.
        # 그리고 상대에게 줄때는

        return np.argmax(self.eval_board) # 1차원 배열 형식으로 변경해 가장 큰 index 반환
    



    
    

