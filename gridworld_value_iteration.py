# Implements the Value Iteration algorithm, as described in Sutton & Barto 2nd Ed, p.83.
import numpy as np
from gridworld_gym_env import ACTION_SPACE, standard_grid, print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9


def calc_optimal_policy(env, Vstar):
    # find the policy that leads to optimal value function
    policy = {}
    for s in env.actions.keys():
        best_a = None
        best_value = float('-inf')
        # loop through all possible actions to find the best current action
        for a in ACTION_SPACE:
            v = 0
            for s2 in env.all_states():
                # reward is a function of (s, a, s'), 0 if not specified
                r = rewards.get((s, a, s2), 0)
                v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * Vstar[s2])

            # best_a is the action associated with best_value
            if v > best_value:
                best_value = v
                best_a = a
        policy[s] = best_a
    return policy



if __name__ == '__main__':
    env = standard_grid()
    transition_probs, rewards = env.get_transition_probs_and_rewards()

    # print rewards
    print("rewards:")
    print_values(env.rewards, env)

    # initialize V(s)
    V = {}
    states = env.all_states()
    for s in states:
        V[s] = 0

    # repeat until convergence
    iter = 0
    policy = {}
    #biggest_value_update = float('inf')
    #while biggest_value_update > SMALL_ENOUGH:
    while True:
        """For each state, calculate
                V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
        ENTER CODE HERE
        """
        delta = 0
        for s in states:
            old_v = V[s]
            new_v = 0
            best_a = None
            for a in ACTION_SPACE:
                v = 0
                for s2 in env.all_states():
                    # reward is a function of (s, a, s'), 0 if not specified
                    r = rewards.get((s, a, s2), 0)
                    v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])
                if v > new_v:
                    new_v = v
                    best_a = a
            policy[s] = best_a
            V[s] = new_v
            delta = max(delta, abs(old_v - V[s]))
        if delta < SMALL_ENOUGH:
            break
        iter += 1

    # use V* to calculate the optimal policy
    #policy = calc_optimal_policy(env, V)

    print("values:")
    print_values(V, env)
    print("policy:")
    print_policy(policy, env)
