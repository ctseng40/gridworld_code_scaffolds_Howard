# Implements the Monte Carlo algorithm (without exploring starts), as described in Sutton & Barto 2nd Ed, p.101.
import operator

import numpy as np

import gridworld_gym_env
from gridworld_gym_env import standard_grid, ACTION_SPACE
from gridworld_gym_env import print_values, print_policy

GAMMA = 0.9
epsilon=0.1

def get_epsilon_greedy_action(state):#, epsilon=0.1):
    """Implement epsilon greedy decision
    ENTER CODE HERE
    """
    action = np.random.choice(ACTION_SPACE, p=policy[state])
    return action


def play_episode(env, policy):
    # returns a list of states, actions, and returns corresponding to the game played
    s = env.reset()
    a = get_epsilon_greedy_action(s)
    #print(a)
    #input()

    # be aware of the timing: each triple is s(t), a(t), r(t)
    # but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t).
    # the first state has no reward since we didn't arrive there by a previous action
    r = 0
    states_actions_rewards = [(s, a, r)]
    done = False
    while not done:
        s, r, done, _ = env.step(a)
        if env.game_over():
            # no further actions are taken from the terminal state
            a = None
        else:
            a = get_epsilon_greedy_action(s)  # the next state is stochastic
        states_actions_rewards.append((s, a, r))

    # calculate the returns by working backwards from the terminal state
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last G, which is meaningless since it doesn't correspond to any move
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA * G
    states_actions_returns.reverse()  # we want it to be in order of state visited
    #print(states_actions_returns)
    #input()
    return states_actions_returns


if __name__ == '__main__':
    env = standard_grid()

    # state -> action
    # initialize a random policy
    policy = {}
    #for s in env.actions.keys():
    #    policy[s] = np.random.choice(ACTION_SPACE)
    n = len(ACTION_SPACE)
    for state in env.actions.keys():
        policy[state] = [1/n for _ in range(n)]

    # initialize Q(s,a) and returns
    Q = {}
    returns = {}  # dictionary of state -> list of returns we've received
    pairs_visited = {}
    states = env.all_states()
    for s in states:
        if s in env.actions:  # not a terminal state
            Q[s] = {}
            for a in ACTION_SPACE:
                Q[s][a] = 0
                returns[(s, a)] = []
                pairs_visited[(s,a)] = 0
        else:
            # terminal state or state we can't otherwise get to
            pass

    # repeat a large number of times until convergence is reached
    deltas = []
    for t in range(50000):
        if t % 1000 == 0:
            print('Iteration %d' % t)
            """generate an episode using the current policy, 
            update the returns for every state-action pair,
            calculate Q for each state-action pair,
            update the policy at every state using pi(s) = argmax[a]{ Q(s,a) }
            ENTER CODE HERE
            """
            deltas = play_episode(env, policy)
            #print(deltas)
            #input()
            #for idt, (state, action, _) in enumerate(deltas):
            #    G = 0
            #    discount = 1
            #    if pairs_visited[(state, action)] == 0:
            #        pairs_visited[(state, action)] += 1
            #        for t, (_, _, reward) in enumerate(deltas[idt:]):
            #            G += reward * discount
            #            discount *= GAMMA
            #            returns[(state, action)].append(G)
            for (state, action, G) in deltas:
                returns[(state, action)].append(G)
            
            for state, action, _ in deltas:
                Q[state][action] = np.mean(returns[(state, action)])
                #update_policy
                actions = [ Q[state][a] for a in ACTION_SPACE]
                a_max = np.argmax(actions)
                #print(a_max)
                #input()
                n_actions = len(ACTION_SPACE)
                probs = np.ones(n_actions) * (epsilon / n_actions)
                probs[a_max] = 1 - epsilon + (epsilon / n_actions)
                policy[state] = probs
            
            for state_action in pairs_visited.keys():
                pairs_visited[state_action] = 0
            
            deltas = []
            
                
    # find the optimal state-value function
    # V(s) = max[a]{ Q(s,a) }
    V = {s: Q[s][ACTION_SPACE[np.argmax(policy[s])]] for s in Q.keys() if s in env.actions.keys()}
    policy_print = {}
    for s in policy.keys():
        policy_print[s]=ACTION_SPACE[np.argmax(policy[s])]

    print("final values:")
    print_values(V, env)
    print("final policy:")
    print_policy(policy_print, env)
