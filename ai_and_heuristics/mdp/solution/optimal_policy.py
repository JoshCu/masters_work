import helpers
from mdp import MDP
import sys

if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)

    success_rate = float(sys.argv[1])
    penalty = float(sys.argv[2])
    m = MDP(success_chance=success_rate, penalty=penalty)

    utility = m.value_iteration_max(m.current_utility)
    p = m.get_optimal_policy(utility)
    helpers.print_grid(utility)
    helpers.print_grid(p, True)
    helpers.assignment_out(p)
