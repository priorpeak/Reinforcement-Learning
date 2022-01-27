
from typing import final
import numpy as np
import matplotlib.pyplot as plt
import random

# discounting rate
gamma = 0.99

#reward outside of a terminal state
rewardSize = -1 

#size of grid
gridSize = 4 

#two terminal states
terminationStates = [[0,0], [gridSize-1, gridSize-1]] 

#four actions depending on where we want to move 
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]] 

#number of iterations we want to do
numIterations = 100


def actionRewardFunction(initialPosition, action):
    #this function returns a reward of rewardSize each step unless you are in a terminal state
    #in that case, it returns a reward of zero
    #it returns  the next state and reward
    
    
    #first check if we are in a termination state 
    #in that case, reward is zero and the position remains the same 
    
    if initialPosition in terminationStates:
        return initialPosition, 0
    
    #if we are not in a termination state, we returne the variable rewardSize 
    reward = rewardSize
    
    #calculcate next position 
    finalPosition = np.array(initialPosition) + np.array(action)
    
    #now check if the next position brings you out of the grid
    #if so, the finalposition should be the same as the initial position
    
    if -1 in finalPosition or gridSize in finalPosition: 
        finalPosition = initialPosition
        
    return finalPosition, reward
    
    

#initialize value map to all zeros 
valueMap = np.zeros((gridSize, gridSize))

#define the state space
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

deltas = []
for it in range(numIterations):
    
    #make a copy of the value function to manipulate during the algorithm
    copyValueMap = np.copy(valueMap)
    
    #this will be set to Vcurrent - Vnext
    deltaState = []
    for state in states:
    
        #the next variable will be equal to the new V iterate by the end of the process
        weightedRewards = 0

        #Bellman Operator
        TV = []
        
        #Compute the Bellman iterate
        for action in actions:
            #compute next position and reward from taking that action
            finalPosition, reward = actionRewardFunction(state, action)
            ### ADD A SINGLE LINE HERE UPDATING THE VARIABLE weightedRewards
            TV.append(reward + gamma * valueMap[finalPosition[0], finalPosition[1]])

        #weightedRewards is max of TV array
        weightedRewards = np.max(TV)
        
        #append Vcurrent-Vnext for the current state
        deltaState.append(np.abs(copyValueMap[state[0], state[1]]-weightedRewards))
        
        #update the value of the next state, but in the copy rather than the original
        copyValueMap[state[0], state[1]] = weightedRewards
        
    #this is now an array of size numIterations, where every entry is an array of Vcurrent-Vmax    
    deltas.append(deltaState)
    
    #update the value map with what we just computed
    valueMap = copyValueMap
    
    #for selected iterations, print the value function
    if it in [0,1,2,9, 99, numIterations-1]:
        print("Iteration {}".format(it+1))
        print(valueMap)
        print("")

directionsArray = ['U', 'D', 'R', 'L']
readableMap = np.empty((4, 4), dtype=np.str_)

for state in states:
    prevReward = -1000000
    for k, action in enumerate(actions):
        #compute next position and reward from taking that action
        finalPosition, reward = actionRewardFunction(state, action)
        reward += valueMap[finalPosition[0], finalPosition[1]]
        if reward > prevReward:
            prevReward = reward
            #Terminal state handling
            if prevReward == 0:
                readableMap[state[0], state[1]] = 'X'
            else:
                readableMap[state[0], state[1]] = directionsArray[k]

# valueMap = copyValueMap
print("Readable optimal policy choice:")
print(readableMap)
        
#plot how the deltas decay        
plt.figure(figsize=(20, 10))
plt.plot(deltas)

#print(deltas)

