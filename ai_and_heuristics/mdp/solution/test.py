import helpers
import copy
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
ROWS = 3
COLUMNS = 4
START_UTILITY = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

DISCOUNT = 0.95
REWARD = [[-0.05, -0.05, -0.05, 1], [-0.05, -0.05, -0.05, -1], [-0.05, -0.05, -0.05, -0.05]]
POLICY = helpers.load_policy("case1.csv")
SUCCESS_CHANCE = .8


def calculate_reward(utility, row, column, action):
    '''
    Perform an action at a state and get the new utility of that action at that state
    '''
    move_row, move_col = ACTIONS[action]
    new_row, new_col = row+move_row, column+move_col
    if new_row < 0 or new_col < 0 or new_row >= ROWS or new_col >= COLUMNS or (
            new_row == new_col == 1):  # collide with the boundary or the wall
        util = utility[row][column]
        reward = REWARD[row][column]
    elif new_row <= 1 and new_col == 3:  # if terminal state
        util = 0
        reward = REWARD[new_row][new_col]
    else:
        util = utility[new_row][new_col]
        reward = REWARD[new_row][new_col]
    return reward + (DISCOUNT * util)


def calculate_utility(utility, row, column, action):
    '''Calculate the utility of a state given an action and chance of failure'''
    slip_chance = (1-SUCCESS_CHANCE)/2
    left = (action-1) % 4
    right = (action+1) % 4
    u = 0
    u += slip_chance * calculate_reward(utility, row, column, left)
    u += SUCCESS_CHANCE * calculate_reward(utility, row, column, action)
    u += slip_chance * calculate_reward(utility, row, column, right)
    return u


def value_iteration():
    utility = copy.deepcopy(START_UTILITY)
    for i in range(20):
        next_utility_grid = copy.deepcopy(START_UTILITY)
        print(next_utility_grid)
        for r in range(ROWS):
            for c in range(COLUMNS):
                if (r <= 1 and c == 3) or (r == c == 1):
                    continue
                next_utility_grid[r][c] = calculate_utility(utility, r, c, POLICY[r][c])
        utility = copy.deepcopy(next_utility_grid)
        print(utility)
        helpers.print_grid(utility)
    return utility


if __name__ == "__main__":
    u = value_iteration()
    print(u)
