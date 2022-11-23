"""
This file calls methods in mdp.py to calculate the optimal policy for a given gridworld
"""

__author__ = 'Josh Cunningham'
__copyright__ = 'Copyright 2022, MDP'
__email__ = 'Josh.Cu@gmail.com'


import helpers
from mdp import MDP
import sys

if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)

    success_rate = float(sys.argv[1])
    penalty = float(sys.argv[2])
    m = MDP(success_chance=success_rate, penalty=penalty)

    utility = m.value_iteration(m.current_utility, True)
    p = m.get_optimal_policy(utility)
    helpers.print_grid(utility)
    helpers.print_grid(p, True)
    helpers.write_policy(helpers.assignment_out(p))
