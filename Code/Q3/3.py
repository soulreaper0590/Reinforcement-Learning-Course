
import gym
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import tqdm
import gym_gridworld
mapFiles = ["map1.txt", "map2.txt", "map3.txt"]
problemis = ["A", "B", "C"]

def policy_fn(theta,x,y,eps):
        A = np.zeros(4, dtype=float) + (eps / 4)
        V = [theta[0]*(x - 1) + theta[1]*(y),theta[0]*(x ) + theta[1]*(y+1),theta[0]*(x +1) + theta[1]*y,theta[0]*(x) + theta[1]*(y+1)]
        # Taking random action if all are same
        if np.allclose(V, np.mean(V)):
            best = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
        else:
            best = np.argmax(V)

        A[best] += (1.0 - eps)
        return A


def find_optimal_policy(Q):
    optimal_policy = defaultdict(lambda: np.zeros(1))
    for k, v in Q.items():
        if np.allclose(v, v[0]):
            optimal_policy[k] = 1
            # optimal_policy[k] = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
        else:
            optimal_policy[k] = np.argmax(v)
    return optimal_policy


def lambda_sarsa(env, p_lambda, num_episodes=1000, iterations=50, gamma=0.9, alpha=0.1, epsilon=0.1):
    env.reset()
    number_of_steps = np.zeros(num_episodes)
    total_return = np.zeros(num_episodes)
    for itr in range(iterations):
        
        print(itr)
        theta = np.ones(2)

        for i_episode in tqdm.tqdm(range(num_episodes)):
            dis_return = 0
            observation = env.reset()
            x = env.state[0]
            y = env.state[1]
            V = [theta[0]*(x - 1) + theta[1]*(y),theta[0]*(x ) + theta[1]*(y+1),theta[0]*(x +1) + theta[1]*y,theta[0]*(x) + theta[1]*(y+1)]
            action_prob = policy_fn(theta,x,y,epsilon)
            a = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            for i in itertools.count():  # Till the end of episode
                # TAKE A STEP
                prev_state = env.state
                next_observation, reward, done, _ = env.step(a)

                env.steps = i
                dis_return += reward  # Updating return
                env.dis_return = dis_return
                x = env.state[0]
                y = env.state[1]
                V = [theta[0]*(x - 1) + theta[1]*(y),theta[0]*(x ) + theta[1]*(y+1),theta[0]*(x +1) + theta[1]*y,theta[0]*(x) + theta[1]*(y+1)]
                #V = [theta[0]*(x-1) + theta[1]*(y),theta[0]*(x + 1) + theta[1]*y,theta[0]*(x - 1) + theta[1]*y,theta[0]*(x) + theta[1]*(y+1)]

                next_action_probs = policy_fn(theta,x,y,epsilon)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs) # Next action
                theta[0] += alpha *(reward + gamma*(theta[0]*(x) + theta[1]*(y)) - theta[0]*(prev_state[0]) + theta[1]*(prev_state[1]) )*(prev_state[0]) 
                theta[1] += alpha *(reward + gamma*(theta[0]*(x) + theta[1]*(y)) - theta[0]*(prev_state[0]) + theta[1]*(prev_state[1]) )*(prev_state[1]) 
                
                if done or i == 10000:
                    env.dis_return = 0
                    env.steps = 0
                    if done:
                        print("ho gaya")
                    break
                observation = next_observation
                a = next_action
            # print("Total steps taken is :", i)
            number_of_steps[i_episode] += i  # Updating Number of steps
            total_return[i_episode] += dis_return  # Updating return

    return number_of_steps/iterations,total_return/iterations





for grids in range(1):
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[grids]
    num_episodes = 30
    iterations = 2
    env.reset()
    env.first_time = False

    sarsa_lambda_values = [ 0.3, 0.5]
    number_of_steps = np.zeros((num_episodes,3))
    total_return = np.zeros((num_episodes,3))
    for l in range(len(sarsa_lambda_values)):
        env.figtitle = " lambda "+str(sarsa_lambda_values[l])
        number_of_steps[:,l],total_return[:,l] = lambda_sarsa(env, num_episodes=num_episodes, iterations=iterations,gamma=0.99, p_lambda=sarsa_lambda_values[l] ,alpha=0.01, epsilon=0.1)
    #for i in range(1):
    plt.plot(number_of_steps[:,1]/iterations)
    #plt.legend((" lambda "+str(sarsa_lambda_values[0])," lambda "+str(sarsa_lambda_values[1])," lambda "+str(sarsa_lambda_values[2])," lambda "+str(sarsa_lambda_values[3])," lambda "+str(sarsa_lambda_values[4])," lambda "+str(sarsa_lambda_values[5])))
    plt.xlabel("num_episodes")
    plt.ylabel("Average_steps")
    plt.title("Limear Function Approximator Averge Steps Taken To reach goal")
    plt.show()
    #for i in range(3):
    plt.plot(number_of_steps[:,1]/iterations)
    plt.xlabel("num_episodes")
    plt.ylabel("Rewards")
    plt.title("linear function approximator averge reward")
    plt.show()