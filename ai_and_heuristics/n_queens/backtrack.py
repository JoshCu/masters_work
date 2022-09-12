"""
This script is designed to solve constrained n-queens using a backtracking algorithm.
All the functions not directly related to backtracking have been moved to the helpers module.
This should make the code much easier to follow.
"""

__author__ = 'Josh Cunningham'
__copyright__ = 'Copyright 2022, constrained n-queens'
__email__ = 'Josh.Cu@gmail.com'

import helpers
import sys
import time

sys.setrecursionlimit(10**9)
# please forgive this global,
# python doesn't have real constants without using something like pydantic
START_TIME = time.time()


def backtrack_grid(grid, frozen_column=-1):
    '''
    Back tracks the grid as far as needed up graph.
    As we are searching by looping through columns left to right and then through rows top to bottom,
    we know that the previous position in the tree can be found by moving the queen in the left column down.
    If the queen cannot be moved down, then it is removed, 
        and we try to move the queen in the column of the left of that down.
    If the queen is in the lowest left most position and needs to be moved, then no solution can be found
    |x|0|0|0|                 |x|0|0|0|
    |0|0|0|0|  backtracks to  |0|0|0|0|
    |0|x|0|0|                 |0|0|0|0|
    |0|0|x|0|                 |0|x|0|0|

    |x|0|0|0|                 |0|0|0|0|
    |0|0|0|0|  backtracks to  |x|0|0|0|
    |0|0|0|0|                 |0|0|0|0|
    |0|x|x|0|                 |0|0|0|0|

    For this to work for constrained N-Queens, we cannot backtrack the column with the first queen in it
    To do this I just save the column number the first queen was in as frozen_column,
    and skip it when looping over the columns
    '''
    rows = helpers.get_grid_width(grid)
    columns = rows
    # loop over columns right to left
    for column_number in range(columns-1, -1, -1):
        if column_number == frozen_column:
            continue
        for row_number in range(0, rows):
            if grid[row_number][column_number] == 1:
                if row_number < rows - 1:
                    # if the queen can be moved down, do that and stop
                    grid[row_number][column_number] = 0
                    grid[row_number + 1][column_number] = 1
                    if helpers.is_safe_all_around(grid, row_number+1, column_number):
                        return
                else:
                    # if not, remove the queen and check the next column
                    grid[row_number][column_number] = 0

    # if we reach this point the grid can't be backtracked
    print("Grid can't be backtracked, no solution found")
    sys.exit(1)


def place_next_queen(grid, frozen_column=-1):

    rows = helpers.get_grid_width(grid)
    columns = rows
    placed = False
    start = helpers.get_first_empty_column(grid)
    for column_number in range(start, columns):
        for row_number in range(0, rows):

            if grid[row_number][column_number] == 0:
                # if no queen, place a queen
                grid[row_number][column_number] = 1
                valid_position = helpers.is_safe_all_around(grid, row_number, column_number)

                if valid_position:
                    # if it fits in the grid then return true
                    return True
                elif not valid_position and row_number == rows - 1:
                    # if not, backtracking and try again
                    backtrack_grid(grid, frozen_column)
                    return place_next_queen(grid, frozen_column)
                else:
                    # if not valid and not on final row, backtrack last guess
                    # by setting value back to zero and looping to the next position
                    grid[row_number][column_number] = 0

    return placed


def search_solutions(grid, frozen_column=-1):
    num_queens = helpers.count_queens(grid)
    grid_width = helpers.get_grid_width(grid)

    # if n queens are placed and the grid is valid
    if (num_queens == grid_width) and (helpers.is_valid(grid)):
        print("Solution Found!")
        helpers.print_grid(grid)
        print('\n')
        helpers.save_grid(grid)
        print("--- %s seconds ---" % (time.time() - START_TIME))
        sys.exit(0)
    elif (num_queens == grid_width) and not (helpers.is_valid(grid)):
        print("no solution found")
        sys.exit(1)

    queen_placed = place_next_queen(grid, frozen_column)

    if num_queens < grid_width and not queen_placed:
        print("no solution found")
        sys.exit(1)

    search_solutions(grid, frozen_column)


if __name__ == "__main__":
    frozen_column = -1

    grid, frozen_column = helpers.load_grid("input.csv")
    # comment out line above and uncomment below if you want to try a grid with no input
    #grid = helpers.init_grid(16)

    helpers.print_grid(grid)
    search_solutions(grid, frozen_column)
