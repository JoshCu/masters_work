def get_grid_width(grid):
    return len(grid.keys())


def print_grid(grid : dict):
    rows = get_grid_width(grid)
    columns = rows
    for column_number in range(0,rows):
        ## separates rows
        print("\n", end = "|")
        for row_number in range(0,columns):
            print(grid[column_number][row_number], end ="|")

def load_grid(filename):
    with open(filename,'r', encoding = 'utf-8') as f:
        line = f.readline().strip().split(',')
        width = len(line)
    # easier to just close and reopen the file than to undo the readline advance
    frozen_column = -1
    with open(filename,'r', encoding = 'utf-8') as f:
        grid = {}
        for column_number in range(0,width):
            line = f.readline().strip().split(',')
            grid[column_number] = {}
            for row_number in range(0,width):
                if (int(line[row_number]) == 1):
                    frozen_column = row_number
                grid[column_number][row_number] = int(line[row_number])
        return grid, frozen_column


def save_grid(grid : dict):
    rows = get_grid_width(grid)
    columns = rows
    with open("solution.csv",'w', encoding = 'utf-8') as f:
        for column_number in range(0,rows):
            row = ""
            for row_number in range(0,columns):
                row += f"{grid[column_number][row_number]},"
            row = row[:-1]
            f.write(row)
            f.write('\n')

def init_grid(width):
    grid = {}
    for column_number in range(0,width):
        grid[column_number] = {}
        for row_number in range(0,width):
            grid[column_number][row_number] = 0
    return grid

def count_queens(grid):
    rows = get_grid_width(grid)
    columns = rows
    queens = 0 
    for column_number in range(0,columns):
        for row_number in range(0,rows):
            queens += grid[column_number][row_number]
    return queens

def __evaluate_columns(grid) -> bool:
    rows = get_grid_width(grid)
    columns = rows
    for column_number in range(0,columns):
        queens = 0 
        for row_number in range(0,rows):
            queens += grid[column_number][row_number]
        if queens > 1:
            return False
    return True

def __evaluate_rows(grid) -> bool:
    rows = get_grid_width(grid)
    columns = rows
    for row_number in range(0,rows):
        queens = 0 
        for column_number in range(0,columns):
            queens += grid[column_number][row_number]
        if queens > 1:
            return False
    return True

def __evaluate_diagonals(grid) -> bool:
    rows = get_grid_width(grid)
    columns = rows
    # calculate left to right diagonals
    # |x|0|0|
    # |0|x|0|
    # |0|0|x|
    for column_number in range(0,columns):
        row = 0
        column = column_number
        queens = 0
        while(row < rows and column < columns):
            queens += grid[column][row]
            row += 1
            column += 1
        if queens > 1:
            return False

    for row_number in range(0,rows):
        row = row_number
        column = 0
        queens = 0
        while(row < rows and column < columns):
            queens += grid[column][row]
            row += 1
            column += 1
        if queens > 1:
            return False

    # calculate right to left diagonals
    # |0|0|x|
    # |0|x|0|
    # |x|0|0|

    for column_number in range(0,columns):
        row = 0
        column = column_number
        queens = 0
        while(row < rows and column >= 0):
            queens += grid[column][row]
            row += 1
            column -= 1
        if queens > 1:
            return False

    for row_number in range(0,rows):
        row = row_number
        column = columns -1
        queens = 0
        while(row < rows and column >= 0):
            queens += grid[column][row]
            row += 1
            column -= 1
        if queens > 1:
            return False

    # if all diagonals don't have more than one queen
    return True


def evaluate_positions(grid) -> bool:
    return __evaluate_columns(grid) and __evaluate_rows(grid) and __evaluate_diagonals(grid)


# My original checker was too expensive so I'm reusing Dr.Duans
# I did ask her if this was ok beforehand 
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
def isValid(grid):
    N = get_grid_width(grid)
    for row in range(N):
        Qs = 0
        for col in range(N):
            if grid[row][col] == 1:
                Qs += 1
                if not isSafe(grid, row, col, N):
                    return False
        if Qs != 1:  # making sure there is a queen per row
            return False

    return True

def get_first_empty_column(grid):
    rows = get_grid_width(grid)
    columns = rows
    for column_number in range(0,columns):
        queens = 0 
        for row_number in range(0,rows):
            queens += grid[row_number][column_number]
        if queens == 0:
            return column_number
    