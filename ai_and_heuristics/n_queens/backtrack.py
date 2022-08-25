import helpers
import sys



def place_next_queen(grid):
    rows = helpers.get_grid_width(grid)
    columns = rows
    queens = 0 
    placed = False
    for column_number in range(0,columns):
        for row_number in range(0,rows):

            if grid[row_number][column_number] == 0:
                grid[row_number][column_number] = 1
                if helpers.evaluate_positions(grid):
                # if it fits in the grid then return true
                    return True
                else:
                # if not, backtrack to old grid before placement
                # by setting value back to zero
                    grid[row_number][column_number] = 0
                    

    return placed


def search_solutions(grid):
    num_queens = helpers.count_queens(grid)
    grid_width = helpers.get_grid_width(grid)
    if num_queens == grid_width:
        print("Solution Found!")
        helpers.print_grid(grid)
        sys.exit(1)
    
    queen_placed = place_next_queen(grid)
    print("--------")
    helpers.print_grid(grid)

    if num_queens < grid_width and not queen_placed:
        print("no solution found")
        sys.exit(1)

    search_solutions(grid)


if __name__ == "__main__":
    grid = helpers.init_grid(4)

    helpers.print_grid(grid)
    search_solutions(grid)