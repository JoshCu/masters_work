import helpers
import copy


class MDP:

    def __init__(self, penalty=-0.04, success_chance=0.8, filename="case0.csv"):
        self.penalty = penalty
        self.discount = 0.95
        self.success_chance = success_chance
        # Set up the initial environment
        self.num_actions = 4
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        self.num_rows = 3
        self.num_columns = 4
        self.start_utility = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.current_utility = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0]]
        self.policy = helpers.load_policy(filename)

    def move_agent(self, utility, row, column, action):
        '''
        Perform an action at a state and get the new utility of that action at that state
        '''

        move_row, move_col = self.actions[action]
        new_row, new_col = row+move_row, column+move_col
        if new_row < 0 or new_col < 0 or new_row >= self.num_rows or new_col >= self.num_columns or (
                new_row == new_col == 1):  # collide with the boundary or the wall
            return self.penalty + self.discount * utility[row][column]
        elif new_row <= 1 and new_col == 3:
            if new_row == 0:
                return 1
            else:
                return -1
            # return utility[new_row][new_col]
        else:
            return self.penalty + self.discount * utility[new_row][new_col]

    def calculate_utility(self, utility, row, column, action):
        '''Calculate the utility of a state given an action and chance of failure'''
        slip_chance = (1-self.success_chance)/2
        u = 0
        u += slip_chance * self.move_agent(utility, row, column, (action-1) % 4)
        u += self.success_chance * self.move_agent(utility, row, column, action)
        u += slip_chance * self.move_agent(utility, row, column, (action+1) % 4)
        return u

    def value_iteration(self, utility):
        for i in range(20):
            next_utility_grid = copy.deepcopy(self.start_utility)
            print(next_utility_grid)
            for r in range(self.num_rows):
                for c in range(self.num_columns):
                    if (r <= 1 and c == 3) or (r == c == 1):
                        continue
                    next_utility_grid[r][c] = self.calculate_utility(utility, r, c, self.policy[r][c])
            utility = copy.deepcopy(next_utility_grid)
            print(utility)
            helpers.print_grid(utility)
        return utility

    def value_iteration_max(self, utility):
        for i in range(20):
            next_utility_grid = self.start_utility
            for row in range(self.num_rows):
                for column in range(self.num_columns):
                    if (row <= 1 and column == 3) or (row == column == 1):
                        continue
                    next_utility_grid[row][column] = max([self.calculate_utility(utility, row, column, action)
                                                          for action in range(self.num_actions)])
            utility = next_utility_grid
            helpers.print_grid(utility)
        return utility

    def get_optimal_policy(self, utility):
        '''Get the optimal policy from utility grid'''
        policy = [[-1, -1, -1, -1] for i in range(self.num_rows)]
        for row in range(self.num_rows):
            for column in range(self.num_columns):
                if (row <= 1 and column == 3) or (row == column == 1):
                    continue
                # Choose the action that maximizes the utility
                best_action, highest_utility = None, -float("inf")
                for action in range(self.num_actions):
                    calculated_utility = self.calculate_utility(utility, row, column, action)
                    if calculated_utility > highest_utility:
                        best_action, highest_utility = action, calculated_utility
                policy[row][column] = best_action
        return policy
