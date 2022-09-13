"""
This file contains a series of functions to assist with the nQueens problem.
As the focus of this lab was the implemntation of backtracking, all other helper functions are located here.

"""

__author__ = 'Josh Cunningham'
__copyright__ = 'Copyright 2022, constrained n-queens'
__email__ = 'Josh.Cu@gmail.com'


def get_grid_width(grid: list):
    return len(grid)


def print_grid(grid: list):
    '''
    Helper function to print the grid out to console
    '''
    rows = get_grid_width(grid)
    columns = rows
    for column_number in range(0, rows):
        # separates rows
        print("\n", end="|")
        for row_number in range(0, columns):
            print(grid[column_number][row_number], end="|")


def load_grid(filename):
    '''
    Helper function to load the input.csv and set the frozen column
    Takes relative filepath and returns grid, frozen_column
    '''
    with open(filename, 'r', encoding='utf-8') as f:
        line = f.readline().strip().split(',')
        width = len(line)
    # easier to just close and reopen the file than to undo the readline advance
    frozen_column = -1
    grid = init_grid(width)
    with open(filename, 'r', encoding='utf-8') as f:
        for column_number in range(0, width):
            line = f.readline().strip().split(',')
            for row_number in range(0, width):
                if (int(line[row_number]) == 1):
                    frozen_column = row_number
                grid[column_number][row_number] = int(line[row_number])
        return grid, frozen_column


def save_grid(grid: list):
    '''
    Helper function to save grid dictionary to a file called solution.csv
    '''
    rows = get_grid_width(grid)
    columns = rows
    with open("solution.csv", 'w', encoding='utf-8') as f:
        for column_number in range(0, rows):
            row = ""
            for row_number in range(0, columns):
                row += f"{grid[column_number][row_number]},"
            row = row[:-1]
            f.write(row)
            f.write('\n')


def init_grid(width):
    '''
    Helper to initialise and empty nested dictionary grid structure
    '''
    list_of_lists = []
    for i in range(width):
        list_of_lists.append([0] * width)
    return list_of_lists

# My original checker was too expensive so I'm reusing Dr.Duans
# I did ask her if this was ok beforehand


def is_safe_all_around(grid, row, col):
    '''
    Helper to check if the queen at a given position is safe
    Because of the constrain we have to check the right side too
    Takes grid, row number, column number, total grid width returns True if safe, else False
    '''
    N = get_grid_width(grid)
    # Check this row
    for i in range(N):
        if i != col:
            if grid[row][i] == 1:
                return False

    # Check upper diagonal on left
    for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
        if grid[i][j] == 1:
            return False

    # Check lower diagonal on left
    for i, j in zip(range(row+1, N, 1), range(col-1, -1, -1)):
        if grid[i][j] == 1:
            return False

    # Check upper diagonal on right
    for i, j in zip(range(row-1, -1, -1), range(col+1, N)):
        if grid[i][j] == 1:
            return False

    # Check lower diagonal on right
    for i, j in zip(range(row+1, N, 1), range(col+1, N)):
        if grid[i][j] == 1:
            return False

    return True


def get_first_empty_column(grid: list):
    '''
    Helper function that returns the first column without a queen from left to right
    Takes grid, returns column index number e.g. first column == 0'''
    rows = get_grid_width(grid)
    columns = rows
    for column_number in range(0, columns):
        queens = 0
        for row_number in range(0, rows):
            queens += grid[row_number][column_number]
        if queens == 0:
            return column_number
