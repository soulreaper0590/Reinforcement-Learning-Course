import numpy as np
from ads import UserAdvert
import matplotlib.pyplot as plt
from pylab import *
ACTION_SIZE = 3
STATE_SIZE = 4
TRAIN_STEPS = 10000  # Change this if needed
LOG_INTERVAL = 10
aplha = 0.4

def learnBandit():
    env = UserAdvert()
    rew_vec = []
    rew_vec.append(0)
    rewards = 0.0
    W = np.ones((4,3))
    Ws = np.ones((4,3))
    counte = np.zeros((3,1))
    for train_step in range(TRAIN_STEPS):
        state = env.getState()
        stateVec = state["stateVec"]
        stateId = state["stateId"]
        # ---- UPDATE code below ------
        # Sample from policy = softmax(stateVec X W) [W learnable params]
        # policy = function (stateVec)
        policy = np.dot(stateVec.transpose() ,W)#be the set of parameters that are learnable
        action = int(np.random.choice(range(3)))
        reward = env.getReward(stateId, action)
        # ----------------------------
        # ---- UPDATE code below ------
        # Update policy using reward
        counte[action] = counte[action] + 1;#no. of a particular action is taken
        if(train_step!=0):
            W[:,action] = W[:,action] -  ((reward - 3.0)*stateVec)*(1 - policy[action])/counte[action]#updating the weights of the action of taken according REINFORCE algorithim
        if train_step % LOG_INTERVAL == 0:
            print("Testing at: " + str(train_step))
            count = 0
            test = UserAdvert()
            for e in range(450):
                teststate = test.getState()
                testV = teststate["stateVec"]
                testI = teststate["stateId"]
                # ---- UPDATE code below ------
                # Policy = function(testV)
                policy = np.dot(testV.transpose(),W)
                exp_p = exp(-policy)
                sums = np.sum(exp_p)
                exp_p = exp_p/sums#softmax policy to determine the action taken 
                # ----------------------------
                act = int(np.random.choice(range(3), p=exp_p))
                reward = test.getReward(testI, act)
                count += (reward/450.0)
            if(count > rewards ):
                Ws = W.copy()
                rewards = count;
            else:
                W = Ws.copy()
            rew_vec.append(count)

    # ---- UPDATE code below ------
    # Plot this rew_vec list
    plt.plot(range(len(rew_vec)-1),rew_vec[1:])
    plt.xlabel("Training Steps*10");plt.ylabel("Average reward");
    plt.title("Answer 5")
    plt.savefig("Answer5")
    plt.show()

if __name__ == '__main__':
    learnBandit()
