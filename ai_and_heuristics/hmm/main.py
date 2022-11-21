import math
import copy
# loop through input
from scipy.stats import norm
# generate probability grid

# mean = sqrt((agentX-tileX)^2+(agentY-tileY)^2)
#probability = norm.pdf(emission, mean, 2/3)

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
            print(grid[i][j], end=' ')
        print()

if __name__ == "__main__":
    file1 = open('stationaryCarReading10.csv', 'r')
    lines = file1.readlines()
    tiles = 10
    time = 20
    counter = 0
    std = 2/3
    grid = initialize_grid(tiles,tiles)

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
                grid[x][y] *= probability
        #normalise grid  
        grid = normalise_grid(grid)
        
        counter +=1

    print_grid(grid)