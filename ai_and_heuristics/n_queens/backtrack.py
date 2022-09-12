import helpers
import sys
import time

sys.setrecursionlimit(10**9)
start_time = time.time()

def backtrack_grid(grid, frozen_column = -1):
    '''
    Back tracks grid as far as needed up graph.
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

    '''
    rows = helpers.get_grid_width(grid)
    columns = rows
    #loop over columns right to left
    for column_number in range(columns-1,-1, -1):
        # if column_number == frozen_column:
        #     continue
        for row_number in range(0,rows):
            if grid[row_number][column_number] == 1:
                if row_number < rows -1:
                    grid[row_number][column_number] = 0
                    grid[row_number + 1][column_number] = 1
                    return
                else:
                    grid[row_number][column_number] = 0

    #TODO this is not a great way of exiting
    #if we reach this point the grid can't be backtracked
    print("Grid can't be backtracked, no solution found")
    sys.exit(1)

def place_next_queen(grid, frozen_column = -1):
    rows = helpers.get_grid_width(grid)
    columns = rows
    placed = False
    start = helpers.get_first_empty_column(grid)
    for column_number in range(start,columns):
        for row_number in range(0,rows):

            if grid[row_number][column_number] == 0:
                grid[row_number][column_number] = 1
                valid_position = helpers.isValid(grid)
                if valid_position:
                # if it fits in the grid then return true
                    return True
                elif not valid_position and row_number == rows -1:
                    backtrack_grid(grid)
                    return place_next_queen(grid)
                else:
                # if not valid and not on final row, backtrack last guess
                # by setting value back to zero
                    grid[row_number][column_number] = 0
                    
    return placed


def search_solutions(grid, frozen_column = -1):
    num_queens = helpers.count_queens(grid)
    grid_width = helpers.get_grid_width(grid)
    if (num_queens == grid_width) and not (helpers.evaluate_positions(grid)):
        print("Solution Found!")
        helpers.print_grid(grid)
        print('\n')
        helpers.save_grid(grid)
        print("--- %s seconds ---" % (time.time() - start_time))
        sys.exit(0)
    elif(num_queens == grid_width) and (helpers.evaluate_positions(grid)):
        print("no solution found")
        sys.exit(1)

    queen_placed = place_next_queen(grid)

    if num_queens < grid_width and not queen_placed:
        print("no solution found")
        sys.exit(1)

    search_solutions(grid)


if __name__ == "__main__":

    #grid = helpers.init_grid(10)
    grid, frozen_column = helpers.load_grid("input.csv")
    print(frozen_column)
    helpers.print_grid(grid)
    search_solutions(grid, frozen_column)
