# Python program to grade n-queens solution
import pandas as pd


def printBoard(board, N):
    for i in range(N):
        for j in range(N-1):
            print(board[i][j], ", ", end='')
        print(board[i][N-1])


# check if a queen at [row][col] is attacked. We need to check only left for queen's safety.
def isSafe(board, row, col, N):
    # Check this row on left side
    for i in range(col):
        if board[row][i] == 1:
            return False

    # Check upper diagonal on left
    for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check lower diagonal on left
    for i, j in zip(range(row+1, N, 1), range(col-1, -1, -1)):
        if board[i][j] == 1:
            return False

    return True


# checking if the solution is valid
def isValid(board, N):
    for row in range(N):
        Qs = 0
        for col in range(N):
            if board[row][col] == 1:
                Qs += 1
                if not isSafe(board, row, col, N):
                    return False
        if Qs != 1:  # making sure there is a queen per row
            print("Queens on row ", row, " invalid.")
            return False

    return True


# checking is solution is valid
df = pd.read_csv("input.csv", header=None)
df2 = pd.read_csv("solution.csv", header=None)
board2 = df2.values.tolist()  # make it a list
printBoard(board2, len(board2))
if not isValid(board2, len(board2)):
    print("solution is ..... invalid")
else:
    print("solution is ..... valid")

print(".......................................................")
# checking if initial condition met
df.mul = df.mul(df2)
if df.mul.sum().sum() == 1:
    print("Yes, initial condition is satisfied!")
else:
    print("NO, initial condition is NOT satisfied!")