
import gym
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import tqdm
import gym_gridworld
from pylab import *
mapFiles = ["map1.txt", "map2.txt", "map3.txt"]
figuretitle = ['Policy Gradient  Problem A',
               'Policy Gradient  Problem B', 'Policy Gradient Problem C']


def optimal_policy(theta_x,theta_y):
    optimal_action = defaultdict(lambda: np.zeros(1))
    for k in range(196):
        x = int(k/14) 
        y = k%14 
        optimal_action[k] = np.argmax(theta_x[int(x),:] + theta_y[int(y),:])
    return optimal_action





def policy_gradient_monte(env, num_episodes=1000, iterations=50, gamma=0.9, alpha=0.1, epsilon=0.1):
    env.reset()
    
    number_of_steps = np.zeros(num_episodes)
    total_return = np.zeros(num_episodes)
    for itr in range(iterations):
        theta_x = np.zeros((14,4))
        theta_y = np.zeros((14,4))
        print(itr)
        for i_episode in tqdm.tqdm(range(num_episodes)):
            dis_return = 0
            
            observation = env.reset()
            # Reset the environment and pick the first action
            action_prob = np.zeros(4)
            for i in range(4):
                x = env.state[0]
                y = env.state[1]
                action_prob[i] = theta_x[x,i] + theta_y[y,i]
            action_prob = exp(action_prob)
            sumy = np.sum(action_prob)
            action_prob = action_prob/sumy
            a = np.random.choice([i for i in range(len(action_prob))], p=action_prob)
            rewards = np.zeros(1000)
            epi_statex = np.zeros(1000)
            epi_statey = np.zeros(1000)
            epi_action = np.zeros(1000)
            steps = 0
            for i in itertools.count():  # Till the end of episode or till the desired end
                # TAKE A STEP
                epi_statex[i] = int(env.state[0])
                epi_statey[i] = int(env.state[1])
                epi_action[i] = int(a)
                next_observation, reward, done, _ = env.step(a)

                env.steps = i
                dis_return += reward  # Updating return
                rewards[i] = reward
                env.dis_return = dis_return
                for r in range(4):
                    x = env.state[0]
                    y = env.state[1]
                    action_prob[r] = theta_x[x,r] + theta_y[y,r]
                action_prob = exp(action_prob)
                sumy = np.sum(action_prob)
                action_prob = action_prob/sumy
                next_action= np.random.choice([i for i in range(len(action_prob))], p=action_prob)

                if done or i == 999:
                    steps = i
                    number_of_steps[i_episode] += steps 
                    total_return[i_episode] += dis_return
                    env.steps = 0 
                    env.dis_return = 0
                    break
                #print(env.state,a,dis_return)
                observation = next_observation
                a = next_action
            for i in range(steps):
                G = 0
                for p in arange(steps - i):
                    G += rewards[i + p]
                #print(epi_statex[i],epi_action[i])
                sumy = np.sum(exp(theta_x[int(epi_statex[i]),:] + theta_y[int(epi_statey[i]),:] ) )
                theta_x[int(epi_statex[i]),int(epi_action[i])] += alpha*(gamma**i)*G*((sumy - 1)/sumy)
                theta_y[int(epi_statey[i]),int(epi_action[i])] += alpha*(gamma**i)*G*((sumy - 1)/sumy)
    plt.plot(number_of_steps/iterations)
    plt.xlabel("num_episodes")
    plt.ylabel("Average_steps")
    plt.title("Policy Gradient Averge Steps Taken To reach goal")
    #plt.figuretitle("Policy Gradient Averge Steps Taken To reach goal")
    plt.show()
    plt.plot(total_return/iterations)
    plt.xlabel("num_episodes")
    plt.ylabel("Rewards")
    plt.title("Policy Gradient Averge Reward per episode")
    #plt.figuretitle("Policy Gradient Averge Reward per episode")
    plt.show()
    return theta_x,theta_y,number_of_steps,total_return




num_episodes = 250
iterations = 10
number_of_steps = np.zeros((num_episodes,3))
total_returns = np.zeros_like(number_of_steps)


for grids in range(3):
    env = gym.make("GridWorld-v0")
    
    env.saveFile = False
    env.mapFile = mapFiles[grids]
    env.figtitle = figuretitle[grids]

    env.reset()
    env.first_time = False
    print(figuretitle[grids])
    theta_x = np.zeros((14,4))
    theta_y = np.zeros((14,4))
    theta_x,theta_y,number_of_steps[:,grids],total_returns[:,grids]  = policy_gradient_monte(env, num_episodes=num_episodes, iterations=iterations,gamma=0.9, alpha=0.01, epsilon=0.1)
    


    optimal_action = optimal_policy(theta_x,theta_y)
    env.optimal_policy = optimal_action
    env.draw_arrows = True
    env.render(mode='human')



plt.plot(number_of_steps[:,0]/iterations)
plt.plot(number_of_steps[:,1]/iterations)
plt.plot(number_of_steps[:,2]/iterations)
plt.legend((figuretitle[0],figuretitle[1],figuretitle[2]))
plt.xlabel("num_episodes")
plt.ylabel("Average_steps")
plt.title("Policy Gradient Steps Taken To reach goal")
plt.savefig("Policy Gradient Steps Taken To reach goal all goals")
plt.show()




plt.plot(total_returns[:,0]/iterations)
plt.plot(total_returns[:,1]/iterations)
plt.plot(total_returns[:,2]/iterations)
plt.legend((figuretitle[0],figuretitle[1],figuretitle[2]))
plt.xlabel("num_episodes")
plt.ylabel("Rewards")
plt.title("Policy Gradient Averge Reward per episode")
plt.savefig("Policy Gradient Averge Reward per episode all goals")
plt.show()