## Credit for the Puddle creation library creation
# https://github.com/niravnb/Reinforcement-learning/tree/master/Q%20Learning%2C%20Sarsa%20and%20Policy%20Gradients/Code/Q1_Q_learning_Sarsa/gym_gridworld/envs

import gym
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import tqdm
import gym_gridworld
mapFiles = ["map1.txt", "map2.txt", "map3.txt"]
problemis = ["A", "B", "C"]
figuretitle = ['Sarsa Problem A',
               'Sarsa  Problem B', 'Sarsa Problem C']
figuretitle2 = ['Sarsa lambda Problem A',
               'Sarsa lambda Problem B', 'Sarsa lambda Problem C']
def epsilon_greedy_policy(Q, eps, nA):
    def policy_fn(observation):
        A = np.zeros(nA, dtype=float) + (eps / nA)

        # Taking random action if all are same
        if np.allclose(Q[observation], Q[observation][0]):
            best = np.random.choice(
                4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
        else:
            best = np.argmax(Q[observation])

        A[best] += (1.0 - eps)
        return A
    return policy_fn

def find_optimal_policy(Q):
    optimal_policy = defaultdict(lambda: np.zeros(1))
    for k, v in Q.items():
        optimal_policy[k] = np.argmax(v)
    return optimal_policy


def sarsa(env, num_episodes=1000, iterations=50, gamma=0.9, alpha=0.1, epsilon=0.1):
    env.reset()
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    number_of_steps = np.zeros(num_episodes)
    total_return = np.zeros(num_episodes)
    for itr in range(iterations):
        
        print(itr)
        Q.clear()

        policy = epsilon_greedy_policy(Q, epsilon, env.n_actions)
        for i_episode in tqdm.tqdm(range(num_episodes)):
            dis_return = 0

    
            # Reset the environment and pick the first action
            observation = env.reset()
            action_prob = policy(observation)
            a = np.random.choice(
                    [i for i in range(len(action_prob))], p=action_prob)

            for i in itertools.count():  # Till the end of episode
                # TAKE A STEP
                next_observation, reward, done, _ = env.step(a)

                env.steps = i
                dis_return += reward  # Updating return
                env.dis_return = dis_return


                next_action_probs = policy(next_observation)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs) # Next action
                Q[observation][a] += alpha * \
                    (reward + gamma*Q[next_observation]
                        [next_action] - Q[observation][a]) # Sarsa update

                if done or i == 10000:
                    # print("Total discounted return is :", dis_return)
                    env.dis_return = 0
                    env.steps = 0
                    break
                
                observation = next_observation
                a = next_action
            # print("Total steps taken is :", i)
            number_of_steps[i_episode] += i  # Updating Number of steps
            total_return[i_episode] += dis_return  # Updating return
    return Q,number_of_steps,total_return


def lambda_sarsa(env, p_lambda, num_episodes=1000, iterations=50, gamma=0.9, alpha=0.1, epsilon=0.1):
    env.reset()
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    elig = defaultdict(lambda: np.zeros(env.n_actions))
    number_of_steps = np.zeros(num_episodes)
    total_return = np.zeros(num_episodes)
    for itr in range(iterations):
        
        print(itr)
        Q.clear()
        elig.clear()
        # for k, _ in Q.items():
        #     Q[k][act] += -9
        policy = epsilon_greedy_policy(Q, epsilon, env.n_actions)
        

        for i_episode in tqdm.tqdm(range(num_episodes)):
            dis_return = 0

    
            # Reset the environment and pick the start state 
            observation = env.reset()
            action_prob = policy(observation)
            a = np.random.choice(
                    [i for i in range(len(action_prob))], p=action_prob)

            for i in itertools.count():  # Till the end of episode
                # TAKE A STEP
                next_observation, reward, done, _ = env.step(a)

                env.steps = i
                dis_return += reward  # Updating return
                env.dis_return = dis_return


                next_action_probs = policy(next_observation)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs) # Next action
                TDE = alpha *(reward + gamma*Q[next_observation][next_action] - Q[observation][a])
                elig[observation][a] += 1 
                for k, _ in Q.items():
                    for act in range(env.n_actions):
                        Q[k][act] += TDE*elig[k][act]
                        elig[k][act] = gamma*p_lambda*elig[k][act]
                
                if done or i == 1000:
                    env.dis_return = 0
                    env.steps = 0
                    break
                observation = next_observation
                a = next_action
            # print("Total steps taken is :", i)
            number_of_steps[i_episode] += i  # Updating Number of steps
            total_return[i_episode] += dis_return  # Updating return

    return Q,number_of_steps/iterations,total_return/iterations


num_episodes = 500
iterations = 50
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
    
    Q,number_of_steps[:,grids],total_returns[:,grids] = sarsa(env, num_episodes=num_episodes, iterations=iterations,gamma=0.9, alpha=0.1, epsilon=0.1)
    


    optimal_policy = find_optimal_policy(Q)
    env.optimal_policy = optimal_policy
    env.draw_arrows = True
    env.render(mode='human')
plt.plot(number_of_steps[:,0]/iterations)
plt.plot(number_of_steps[:,1]/iterations)
plt.plot(number_of_steps[:,2]/iterations)
plt.legend((figuretitle[0],figuretitle[1],figuretitle[2]))
plt.xlabel("num_episodes")
plt.ylabel("Average_steps")
plt.title("Sarsa Averge Steps Taken To reach goal")
plt.show()




plt.plot(total_returns[:,0]/iterations)
plt.plot(total_returns[:,1]/iterations)
plt.plot(total_returns[:,2]/iterations)
plt.legend((figuretitle[0],figuretitle[1],figuretitle[2]))
plt.xlabel("num_episodes")
plt.ylabel("Rewards")
plt.title("Sarsa Averge Reward per episode")
plt.show()



for grids in range(3):
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[grids]
    num_episodes = 30
    iterations = 25
    env.reset()
    env.first_time = False

    sarsa_lambda_values = [0, 0.3, 0.5, 0.9, 0.99, 1.0]
    number_of_steps = np.zeros((num_episodes,6))
    total_return = np.zeros((num_episodes,6))
    for l in range(len(sarsa_lambda_values)):
        env.figtitle = figuretitle[grids]+" lambda "+str(sarsa_lambda_values[l])
        Q,number_of_steps[:,l],total_return[:,l] = lambda_sarsa(env, num_episodes=num_episodes, iterations=iterations,gamma=0.9, p_lambda=sarsa_lambda_values[l] ,alpha=0.1, epsilon=0.1)
        optimal_policy = find_optimal_policy(Q)
        env.optimal_policy = optimal_policy
        env.draw_arrows = True
        env.render(mode='human')
    for i in range(6):
        plt.plot(number_of_steps[:,i])
    plt.legend((" lambda "+str(sarsa_lambda_values[0])," lambda "+str(sarsa_lambda_values[1])," lambda "+str(sarsa_lambda_values[2])," lambda "+str(sarsa_lambda_values[3])," lambda "+str(sarsa_lambda_values[4])," lambda "+str(sarsa_lambda_values[5])))
    plt.xlabel("num_episodes")
    plt.ylabel("Average_steps")
    plt.title("Sarsa lambda Averge Steps Taken To reach goal")
    plt.show()
    for i in range(6):
        plt.plot(total_return[:,i])
    plt.legend((" lambda "+str(sarsa_lambda_values[0])," lambda "+str(sarsa_lambda_values[1])," lambda "+str(sarsa_lambda_values[2])," lambda "+str(sarsa_lambda_values[3])," lambda "+str(sarsa_lambda_values[4])," lambda "+str(sarsa_lambda_values[5])))
    plt.xlabel("num_episodes")
    plt.ylabel("Rewards")
    plt.title("Sarsa lambda Averge Rewards per episode")
    plt.show()