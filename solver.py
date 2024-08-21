import numpy as np
import pandas as pd
from typing import List, Union
from tqdm import tqdm

def is_valid(board, row, col, num):
    # Check if the number is already in the row
    for x in range(9):
        if board[row][x] == num:
            return False

    # Check if the number is already in the column
    for x in range(9):
        if board[x][col] == num:
            return False

    # Check if the number is already in the 3x3 box
    start_row = row - row % 3
    start_col = col - col % 3
    for i in range(3):
        for j in range(3):
            if board[i + start_row][j + start_col] == num:
                return False

    return True

def find_empty(board):
    # Find an empty cell in the board
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)

    return None

def solve_sudoku(board):
    # Get the next empty cell
    empty = find_empty(board)
    if not empty:
        return True

    # Get the row and column of the empty cell
    row, col = empty

    # Try to fill the empty cell with a number
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num

            if solve_sudoku(board):
                return True

            board[row][col] = 0

    return False


def board_conversion(board: Union[List, str]) -> Union[List, str]:
    if isinstance(board, str):
        board = np.array([int(digit) for digit in board]).reshape(9, 9)
        return board
    else:
        return "".join([str(digit) for row in board for digit in row])

if __name__ == "__main__":
    data = pd.read_csv("data/sudoku_small.csv")
    data = data.sample(10000)
    
    sudokus = data['quizzes'].values
    solutions = data['solutions'].values
    
    correct = 0
    total = len(sudokus)
    
    for sudoku, solution in tqdm(zip(sudokus, solutions), total=total, desc="Solving Sudokus", unit="sudoku"):
        board = board_conversion(sudoku)
        solve_sudoku(board)
        
        solved = board_conversion(board)
        
        if solved == solution:
            correct += 1
    
    print(f"Accuracy: {correct / total * 100:.2f}%")    
