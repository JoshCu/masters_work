import math
import copy
from scipy.stats import norm

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
    converted_y = tiles - max_x - 1
    print("max probability is at: ", converted_x, converted_y)

if __name__ == "__main__":

    # read input from command line arguments
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 HMMpart1.py <sensor_reading_file> <time_steps>")
        exit()
    sensor_reading_file = sys.argv[1]
    steps = int(sys.argv[2])

    file = open(sensor_reading_file, 'r')
    lines = file.readlines()
    # todo read in tiles
    tiles = int(lines[1].split(',')[3])
    time = steps
    counter = 0
    std = 2/3
    grid = initialise_grid(tiles,tiles)

    for line in lines[1:]:
        if counter >= time:
            break
        
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

    print_guess(grid)
    filename = f"pMap_atTime{time}.csv"
    output_grid_to_file(grid, filename)

