import math
import random

cumReward = dict()
cumReward[('A', 1)] = 16
cumReward[('A', 2)] = 16
cumReward[('B', 1)] = 16

states = []
actions = []

gamma = 1/2

def qCalc(step, state, action):
    alpha = 1 / math.sqrt(step)
    if state == 'A' and action == 1:
        nextState = 'A'
        maxReward = max(cumReward[(nextState, 1)], cumReward[(nextState, 2)])
        reward = 1
    elif state == 'A' and action == 2:
        nextState = 'B'
        maxReward = cumReward[(nextState, 1)]
        reward = 0
    else:
        nextState = 'B'
        maxReward = cumReward[(nextState, 1)]
        reward = 2

    cumReward[(state, action)] += alpha * (reward + gamma * maxReward - cumReward[(state, action)])

    print(cumReward)

def qLearn():
    step = 1
    while step < 1000:
        state = 'A'
        i = 0
        while i < 2:
            i += 1
            state = random.choice(['A', 'B']) if state == 'A' else 'B'
            action = random.choice([1, 2]) if state == 'A' else 1
            qCalc(step, state, action)
            step += 1

qLearn()