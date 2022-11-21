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
            x = int(float(line.split(',')[0]))
            y = int(float(line.split(',')[1]))
            up = float(line.split(',')[2])
            right = float(line.split(',')[3])
            down = float(line.split(',')[4])
            left = float(line.split(',')[5])
            transition_matrix[x][y][0] = up
            transition_matrix[x][y][1] = down
            transition_matrix[x][y][2] = left
            transition_matrix[x][y][3] = right
    return transition_matrix


def initialize_grid(width, height):
    # initialize grid with uniform probability
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
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                f.write(str(grid[i][j]) + ' ')
            f.write('\n')

def print_grid(grid):
    # print grid
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            print(to_precision(grid[i][j],4), end=' ')
        print()

def calculate_new_probability(grid, trans_m, X, Y, probability, grid_size):
    # calculate new probability
    # get each surrounding tile
    # calculate the probability of the agent transitioning from the surrounding tile to the current tile
    # multiply the probability of the agent transitioning from the surrounding tile to the current tile by the probability of the agent being in the surrounding tile
    # add the probabilities of the agent being in the surrounding tiles to get the new probability of the agent being in the current tile

    # wrap around if the agent is at the edge of the grid using grid size
    X_up = (X - 1) % grid_size
    X_down = (X + 1) % grid_size
    Y_left = (Y - 1) % grid_size
    Y_right = (Y + 1) % grid_size
    
    up = grid[X_up][Y]
    down = grid[X_down][Y]
    left = grid[X][Y_left]
    right = grid[X][Y_right]

    # calculate the probability of the agent transitioning from the surrounding tile to the current
    return (up*trans_m[X_up][Y][0] + down*trans_m[X_down][Y][1] + left*trans_m[X][Y_left][2] + right*trans_m[X][Y_right][3])*probability


if __name__ == "__main__":
    file1 = open('movingCarReading10.csv', 'r')
    lines = file1.readlines()
    tiles = 10
    time = 20
    counter = 0
    std = 2/3
    grid = initialize_grid(tiles,tiles)
    transition_matrix = initialise_transition_matrix(tiles)
    last_grid = copy.deepcopy(grid)
    for line in lines[1:]:
        if counter >= time:
            break
        print("new round")
        agentX = int(line.split(',')[0])
        agentY = int(line.split(',')[1])
        emission = float(line.split(',')[2])

        for x in range(tiles):
            for y in range(tiles):
                distance = math.sqrt(math.pow(agentX-x,2)+ math.pow(agentY-y,2))
                probability = norm.pdf(emission, distance, std)
                grid[x][y] = calculate_new_probability(last_grid, transition_matrix, x, y, probability, tiles)
        #normalise grid  
        grid = normalise_grid(grid)

        last_grid = copy.deepcopy(grid)
        
        counter +=1

    print_grid(grid)
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

    print("max probability is at: ", max_x, max_y)



