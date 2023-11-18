import math
import copy
from scipy.stats import norm

def initialise_transition_matrix(tiles, filename = "transitionProb10.csv"):
    # create a 3d array of size tiles*tiles*4
    # 4 is the number of possible moves 
    # 0 = up, 1 = down, 2 = left, 3 = right
    transition_matrix = []
    for i in range(tiles):
        transition_matrix.append([])
        for j in range(tiles):
            transition_matrix[i].append([])
            for k in range(4):
                transition_matrix[i][j].append(0)
    #populate the transition matrix from file
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            row = tiles - int(float(line.split(',')[1])) -1
            col = int(float(line.split(',')[0]))
            up = float(line.split(',')[4])
            right = float(line.split(',')[3])
            down = float(line.split(',')[2])
            left = float(line.split(',')[5])
            transition_matrix[row][col][0] = up
            transition_matrix[row][col][1] = down
            transition_matrix[row][col][2] = left
            transition_matrix[row][col][3] = right
    return transition_matrix


def initialise_grid(width, height):
    # initialise grid with uniform probability
    grid = []
    for i in range(height):
        grid.append([])
        for j in range(width):
            grid[i].append(1/(width*height))            
    return grid    

def normalise_grid(grid):
    # normalise grid
    total = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            total += grid[i][j]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            grid[i][j] /= total
    return grid

def output_grid_to_file(grid, filename):
    # output grid to file
    with open(filename, 'w') as f:
        numbers = ""
        for i in range(len(grid)):
            numbers += f"{i+1},"
        f.write(numbers[:-1])
        f.write('\n')
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                f.write(str(grid[i][j]) + ',')
            f.write('\n')

def print_grid(grid):
    # print grid
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            print(grid[i][j], end=' ')
        print()

def calculate_new_probability(grid, trans_m, Row, Col, probability, grid_size):
    # calculate new probability get each surrounding tile
    # calculate the probability of the agent transitioning from the surrounding tile to the current tile
    # multiply that  by the probability of the agent being in the surrounding tile
    # add all of these together and multiply by the probability of the agent being in the current tile

    # wrap around if the agent is at the edge of the grid using grid size
    Row_up = (Row - 1) % grid_size
    Row_down = (Row + 1) % grid_size
    Col_left = (Col - 1) % grid_size
    Col_right = (Col + 1) % grid_size
    
    above = grid[Row_up][Col]
    below = grid[Row_down][Col]
    left = grid[Row][Col_left]
    right = grid[Row][Col_right]

    return (above*trans_m[Row_up][Col][1] + below*trans_m[Row_down][Col][0] + left*trans_m[Row][Col_left][3] + right*trans_m[Row][Col_right][2])*probability

def print_guess(grid):
    # print grid location of highest probability
    max = 0
    max_x = 0
    max_y = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] > max:
                max = grid[i][j]
                max_x = i
                max_y = j
    # convert this to the "correct" coordinate system
    # (0,0) is bottom left, not top left
    # pandas is backwards and I will die on this hill
    converted_x = max_y
    converted_y = (tiles - max_x) - 1
    print("max probability is at: ", converted_x, converted_y)

if __name__ == "__main__":

# read input from command line arguments
    import sys
    if len(sys.argv) != 4:
        print("Usage: python3 HMMpart2.py <sensor_reading_file> <transition_matrix_file> <time_steps>")
        exit()
    sensor_reading_file = sys.argv[1]
    transition_matrix_file = sys.argv[2]
    steps = int(sys.argv[3]) -1
    # sensor_reading_file = "movingCarReading10.csv"
    # transition_matrix_file = "transitionProb10.csv"
    # steps = 20

    file = open(sensor_reading_file, 'r')
    lines = file.readlines()
    # todo read in tiles
    tiles = int(lines[1].split(',')[3])
    time = steps
    counter = 0
    std = 0.1
    grid = initialise_grid(tiles,tiles)

    transition_matrix = initialise_transition_matrix(tiles, transition_matrix_file)
    last_grid = copy.deepcopy(grid)

    for line in lines[1:]:
        if counter >= time:
            break
        agentX = int(line.split(',')[0])
        agentY = int(line.split(',')[1])
        emission = float(line.split(',')[2])
        agentRow = tiles - agentY - 1
        agentCol = agentX

        for row in range(tiles):
            for col in range(tiles):
                distance = math.sqrt(math.pow(agentRow-row,2)+ math.pow(agentCol-col,2))
                probability = norm.pdf(emission, distance, std)
                current_tile_probability = grid[row][col] * probability
                grid[row][col] = calculate_new_probability(last_grid, transition_matrix, row, col, current_tile_probability, tiles)
        #normalise grid  
        grid = normalise_grid(grid)

        last_grid = copy.deepcopy(grid)
        
        counter +=1

    # rotate grid 90 degrees clockwise 
    #grid = list(zip(*grid[::-1]))

    print_grid(grid)

    print_guess(grid)

    filename = f"pMap_atTime{time}.csv"
    output_grid_to_file(grid, filename)


