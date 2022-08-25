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