import helpers
import sys
from mdp import MDP

if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)

    penalty = float(sys.argv[1])
    filename = sys.argv[2]
    m = MDP(filename=filename, penalty=penalty)

    utility = m.value_iteration(([0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]))
    helpers.print_grid(utility)
    helpers.print_grid(m.policy, True)
    print("UTILITY")
    print(utility[2][0])
