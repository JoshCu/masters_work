"""
This script is designed to solve constrained n-queens using a backtracking algorithm.
All the functions not directly related to backtracking have been moved to the helpers module.
This should make the code much easier to follow.

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

"""

__author__ = 'Josh Cunningham'
__copyright__ = 'Copyright 2022, constrained n-queens'
__email__ = 'Josh.Cu@gmail.com'

import helpers
import time


def search_solutions(grid, current_column, frozen_column):

    width = helpers.get_grid_width(grid)

    # Exit if all queens are placed
    if current_column >= width:
        return True
    # Skip the column the input queen was on

    # Try every row one at a time
    for row in range(width):

        if helpers.is_safe_all_around(grid, row, current_column):
            # if it's not the frozen column then place a queen
            if current_column != frozen_column:
                grid[row][current_column] = 1

            # recurse to progress to the next row and try to place a queen
            # this still needs to be called on the frozen column to continue the chain
            if search_solutions(grid, current_column + 1, frozen_column) == True:
                return True
            # if at some point in the recursion a queen can't be placed,
            # the column to the right will return false and exit up a level to here
            # this will unplace the queen and begin the next loop
            # which is the equivalent of moving the queen down one
            if current_column != frozen_column:
                grid[row][current_column] = 0

    return False


if __name__ == "__main__":
    frozen_column = -2
    start_time = time.time()
    #grid, frozen_column = helpers.load_grid("input.csv")
    # comment out line above and uncomment below if you want to try a grid with no input
    grid = helpers.init_grid(15)

    helpers.print_grid(grid)

    if search_solutions(grid, 0, frozen_column):
        print("Solution Found!")
        helpers.print_grid(grid)
        helpers.save_grid(grid)
        print('\n')
    else:
        print("No Solution :(")
    print("--- %s seconds ---" % (time.time() - start_time))
