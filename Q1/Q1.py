# For learning options optimal policy


import numpy as np
import gym
import gym_gridworld
import itertools
from collections import defaultdict
import sys
from gym import wrappers
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tqdm import tqdm


# Creates epsilon greedy policy
def epsilon_greedy_policyA(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA

        # Taking random action if all are same
        if np.allclose(Q[observation], Q[observation][0]):
            best_action = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
        else:
            best_action = np.argmax(Q[observation])

        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn
#policy Function for the combination of action and options
def epsilon_greedy_policyAO(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA

        # Taking random action if all are same
        if np.allclose(Q[observation], Q[observation][0]):
            best_action = np.random.choice(
                6, 1, p=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])[0]
        else:
            best_action = np.argmax(Q[observation])

        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn
#policy Function just for the options
def epsilon_greedy_policyO(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA

        # Taking random action if all are same
        if np.allclose(Q[observation], Q[observation][0]):
            best_action = np.random.choice(
                2, 1, p=[1/2,1/2])[0]
        else:
            best_action = np.argmax(Q[observation])

        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def SMDPq_learning(env, num_episodes=500, iterations=50, gamma=0.9, alpha=0.1, epsilon=0.1):
    env.reset()
    env.epsilon = epsilon
    env.gamma = gamma
    env.alpha = alpha

    # env.render(mode='human')
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    number_of_steps = np.zeros(num_episodes)
    total_return = np.zeros(num_episodes)
    total_return_steps = np.zeros(num_episodes)

    for itr in range(iterations):
        #print(f"iteration going on:")
        Q.clear()
        Q[env.terminal_state] = np.ones(env.n_actions)

        policy = epsilon_greedy_policyAO(Q, epsilon, env.n_actions)

        for i_episode in tqdm(range(num_episodes)):
            dis_return = 0
            steps_per_episode = 0

            observation = env.reset()  # Start state
            if Question == 1:
                env.state = 0
                env.start_state = 0
            else:
                env.state = 90
                env.start_state = 90

            for i in itertools.count():  # Till the end of episode
                action_prob = policy(observation)
                a = np.random.choice([i for i in range(len(action_prob))], p=action_prob)  # Action selection

                if a > 3:
                    # Passing option policy to enviroment
                    env.options_poilcy = optimal_policy[:,a-4]
                    next_observation, reward, done, _ = env.step( a,optimal_policy)  
                else:
                    next_observation, reward, done, _ = env.step(a,optimal_policy) 

                env.steps = i
                dis_return += reward*gamma**i  # Updating return
                env.dis_return = dis_return
                steps_per_episode += env.options_length

                # Finding next best action from next state
                best_next_a = np.argmax(Q[next_observation])
                # Q Learning update
                Q[observation][a] += alpha*(reward + (gamma**env.options_length)* Q[next_observation][best_next_a] - Q[observation][a])

                if done:
                    env.dis_return = 0
                    env.steps = 0
                    break

                observation = next_observation
            #print("Total steps taken is :", steps_per_episode)
            # Updating Number of steps
            number_of_steps[i_episode] += steps_per_episode
            # Updating return
            total_return[i_episode] += dis_return  # gamma**steps_per_episode
            total_return_steps[i_episode] += gamma**steps_per_episode

    number_of_steps /= iterations
    total_return /= iterations  # Updating return
    total_return_steps /= iterations

    return Q, number_of_steps, total_return, total_return_steps
def Option_q_learning(env, num_episodes=500, iterations=50, gamma=0.9, alpha=0.1, epsilon=0.1):
    env.reset()
    env.epsilon = epsilon
    env.gamma = gamma
    env.alpha = alpha

    # env.render(mode='human')
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    number_of_steps = np.zeros(num_episodes)
    total_return = np.zeros(num_episodes)
    total_return_steps = np.zeros(num_episodes)

    for itr in range(iterations):
        #print(f"iteration going on:")
        Q.clear()
        Q[env.terminal_state] = np.ones(env.n_actions)

        policy = epsilon_greedy_policyO(Q, epsilon, env.n_actions)

        for i_episode in tqdm(range(num_episodes)):
            dis_return = 0
            steps_per_episode = 0

            observation = env.reset()  # Start state
            if Question == 1:
                env.state = 0
                env.start_state = 0
            else:
                env.state = 90
                env.start_state = 90

            for i in itertools.count():  # Till the end of episode
                action_prob = policy(observation)
                a = np.random.choice([i for i in range(len(action_prob))], p=action_prob)
                #print(a)  # Action selection
                env.options_poilcy = optimal_policy[:,a]
                next_observation, reward, done, _ = env.step( a+4,optimal_policy)  
                # print(observation,next_observation)
                # env.options_poilcy = optimal_policy[:,a]
                # next_observation, reward, done, _ = env.step( a,optimal_policy)  
                # print(observation,next_observation)
                env.steps = i
                dis_return += reward*gamma**i  # Updating return
                env.dis_return = dis_return
                steps_per_episode += env.options_length

                # Finding next best action from next state
                best_next_a = np.argmax(Q[next_observation])
                # Q Learning update
                Q[observation][a] += alpha*(reward + (gamma**env.options_length)* Q[next_observation][best_next_a] - Q[observation][a])

                if done :
                    env.dis_return = 0
                    env.steps = 0
                    break
                if i_episode > 100 and steps_per_episode > 1000:
                    env.dis_return = 0
                    env.steps = 0
                    break
                observation = next_observation
            #print("Total steps taken is :", steps_per_episode)
            # Updating Number of steps
            number_of_steps[i_episode] += steps_per_episode
            # Updating return
            total_return[i_episode] += dis_return  # gamma**steps_per_episode
            total_return_steps[i_episode] += gamma**steps_per_episode

    number_of_steps /= iterations
    total_return /= iterations  # Updating return
    total_return_steps /= iterations

    return Q, number_of_steps, total_return, total_return_steps

def option_learning(env, num_episodes=500, iterations=50, gamma=0.9, alpha=0.1, epsilon=0.1):
    env.reset()
    env.epsilon = epsilon
    env.gamma = gamma
    env.alpha = alpha

    Q = defaultdict(lambda: np.zeros(env.n_actions))

    number_of_steps = np.zeros(num_episodes)
    total_return = np.zeros(num_episodes)

    for itr in range(iterations):

        Q.clear()

        policy = epsilon_greedy_policyA(Q, epsilon, env.n_actions)
        figcount = 0
        options_goal = [[25, 56, 77, 103], [103, 25, 56, 77]]
        options_start_states = [[np.append(np.arange(0, 25, 1), 103), np.arange(25, 56, 1), np.arange(56, 77, 1), np.arange(77, 103, 1)],
                                [np.arange(0, 26, 1), np.arange(
                                    26, 57, 1), np.arange(57, 78, 1), np.arange(78, 104, 1)]]

        for op in np.arange(4):

            for i_episode in tqdm(range(num_episodes)):
                dis_return = 0
                steps_per_episode = 0

                observation = env.reset()  # Start state
                env.state = np.random.choice(options_start_states[Option][op])
                env.start_state = env.state
                env.terminal_state = options_goal[Option][op]
                observation = env.state

                for i in itertools.count():  # Till the end of episode
                    action_prob = policy(observation)
                    a = np.random.choice(
                        [i for i in range(len(action_prob))], p=action_prob)  # Action selection

                    
                    next_observation, reward, done, _ = env.step(a,optimal_policy)  # Taking action

                    env.steps = i
                    dis_return += reward*gamma**i  # Updating return
                    env.dis_return = dis_return
                    steps_per_episode += env.options_length

                    if (next_observation not in options_start_states[Option][op]) and (next_observation != options_goal[Option][op]) and not done:
                        # print("Outside the room")
                        break

                    # Finding next best action from next state
                    best_next_a = np.argmax(Q[next_observation])
                    # Q Learning update
                    Q[observation][a] += alpha*(reward + gamma
                                                * Q[next_observation][best_next_a] - Q[observation][a])

                    if done:
                        env.dis_return = 0
                        env.steps = 0
                        break

                    


                    observation = next_observation
            # print("Total steps taken is :", steps_per_episode)
            # Updating Number of steps
            number_of_steps[i_episode] += steps_per_episode
            # Updating return
            total_return[i_episode] += gamma**steps_per_episode

    number_of_steps /= iterations
    total_return /= iterations  # Updating return

    return Q, number_of_steps, total_return

def q_learning(env, num_episodes=500, iterations=50, gamma=0.9, alpha=0.1, epsilon=0.1):
    env.reset()
    env.epsilon = epsilon
    env.gamma = gamma
    env.alpha = alpha

    # env.render(mode='human')
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    Q[env.terminal_state] = np.ones(env.n_actions)

    number_of_steps = np.zeros(num_episodes)
    total_return = np.zeros(num_episodes)
    total_return_steps = np.zeros(num_episodes)

    for itr in range(iterations):

        Q.clear()
        Q[env.terminal_state] = np.ones(env.n_actions)

        policy = epsilon_greedy_policyA(Q, epsilon, env.n_actions)
        figcount = 0

        for i_episode in tqdm(range(num_episodes)):
            dis_return = 0
            steps_per_episode = 0

            observation = env.reset()  # Start state
            if Question == 1:
                env.state = 0
                env.start_state = 0
            else:
                env.state = 90
                env.start_state = 90

            for i in itertools.count():  # Till the end of episode
                action_prob = policy(observation)
                a = np.random.choice(
                    [i for i in range(len(action_prob) )], p=action_prob)  # Action selection

                next_observation, reward, done, _ = env.step(a,optimal_policy = optimal_policy)  # Taking action

                env.steps = i
                dis_return += reward*gamma**i  # Updating return
                env.dis_return = dis_return
                steps_per_episode += env.options_length

                # Finding next best action from next state
                best_next_a = np.argmax(Q[next_observation])
                # Q Learning update
                Q[observation][a] += alpha*(reward + (gamma)* Q[next_observation][best_next_a] - Q[observation][a])

                if done:
                    env.dis_return = 0
                    env.steps = 0
                    break

                observation = next_observation
            #print("Total steps taken is :", steps_per_episode)
            # Updating Number of steps
            number_of_steps[i_episode] += steps_per_episode
            # Updating return
            total_return[i_episode] += dis_return  # gamma**steps_per_episode
            total_return_steps[i_episode] += gamma**steps_per_episode

    number_of_steps /= iterations
    total_return /= iterations  # Updating return
    total_return_steps /= iterations

    return Q, number_of_steps, total_return, total_return_steps

def Intra_Option_q_learning(env, num_episodes=500, iterations=50, gamma=0.9, alpha=0.1, epsilon=0.1):
    env.reset()
    env.epsilon = epsilon
    env.gamma = gamma
    env.alpha = alpha

    Q = defaultdict(lambda: np.zeros(env.n_actions))
    Q[env.terminal_state] = np.ones(env.n_actions)
    Q_hat = defaultdict(lambda: np.zeros(env.n_actions))

    number_of_steps = np.zeros(num_episodes)
    total_return = np.zeros(num_episodes)
    total_return_steps = np.zeros(num_episodes)


    for itr in range(iterations):

        Q.clear()
        Q_hat.clear()
        alpha = alpha
        Q[env.terminal_state] = np.ones(env.n_actions)
        Q_hat[env.terminal_state] = np.ones(env.n_actions)

        policy = epsilon_greedy_policyAO(Q, epsilon, env.n_actions)
        figcount = 0

        for i_episode in tqdm(range(num_episodes)):
            dis_return = 0
            alpha = alpha*.75
            steps_per_episode = 0
            observation = env.reset()  # Start state
            if Question == 1: # Setting start state according to Question
                env.state = 0
                env.start_state = 0
            else:
                env.state = 90
                env.start_state = 90

            for i in itertools.count():  # Till the end of episode
                action_prob = policy(observation)
                a = np.random.choice(
                    [i for i in range(len(action_prob))], p=action_prob)  # Action selection

                env.options_Q = Q # passing Q values to environment
                env.options_Q_hat = Q_hat  # passing Q_hat values to environment
                if a > 3:
                    # passing option optimal policy to environment
                    env.options_poilcy = optimal_policy[:,a-4]
                next_observation, reward, done, _ = env.step(a,optimal_policy = optimal_policy)  # Taking option and also updating Q and Q_hat values accordingly
                Q = env.options_Q # Replacing updated Q values
                Q_hat = env.options_Q_hat # Replacing updated Q_hat values
                


                env.steps = i
                dis_return += reward*gamma**i  # Updating return
                env.dis_return = dis_return
                steps_per_episode += 1
                if done:
                    env.dis_return = 0
                    env.steps = 0
                    break

                observation = next_observation
            # Updating Number of steps
            number_of_steps[i_episode] += steps_per_episode
            # Updating return
            total_return[i_episode] += dis_return
            total_return_steps[i_episode] += gamma**steps_per_episode

    number_of_steps /= iterations
    total_return /= iterations  # Updating return
    total_return_steps /= iterations

    return Q, number_of_steps, total_return, total_return_steps

# Setting some values for Option 1 & Option 2
Option = 1
if Option == 1:
    mapFiles = ["map1.txt", "map2.txt"]
    problemis = ["G1", "G2"]
    figtitle = ["O1", "O2"]

else:
    mapFiles = ["map3.txt", "map4.txt"]
    problemis = ["G1", "G2"]
    figtitle = ["O1", "O2"]

num_episodes = 1000
iterations = 1
optimal_policy = np.zeros((105,4),dtype='int8')# For two diffrent optimal policies
for pb in range(2):
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[0]
    env.n_actions = 4
    env.reset()
    env.first_time = False
    env.draw_circles = True
    env.figtitle = figtitle[pb]
    Option = pb

    Q, number_of_steps, total_return = option_learning(env, num_episodes=num_episodes,iterations=iterations, gamma=0.9, alpha=0.1, epsilon=0.1)
    V = np.zeros(len(Q)) # For storing V values

    for j in range(len(Q)):
        V[j] = np.max(Q[j])
        optimal_policy[j,pb] = np.argmax(Q[j])
    V = preprocessing.minmax_scale(V, feature_range=(0, 0.5))
    for g in [25, 56, 77, 103]:
        V[g] = 0
    if pb == 0: # Setting optimal action for hallway states according to option
        optimal_policy[25,pb] = 1
        optimal_policy[56,pb] = 2
        optimal_policy[77,pb] = 3
        optimal_policy[103,pb] = 0
    else:
        optimal_policy[25,pb] = 3
        optimal_policy[56,pb] = 0
        optimal_policy[77,pb] = 1
        optimal_policy[103,pb] = 2
    # if Question != 1:
    #     env.state = 90
    #     env.start_state = 90
    # env.V = V # passing value function to environment 
    # env.optimal_policy = optimal_policy[:,pb] # passing optimal policy to environment
    # env.terminal_state = 1000
    # env.draw_circles = True
    # env.draw_arrows = False
    # env.figtitle = figtitle[pb]+'_'+str(num_episodes)
    # env.render(mode='human')

    # env.draw_circles = False
    # env.draw_arrows = True
    # env.render(mode='human')
iterations = 10
num_episodes = 1000
for pb in range(1):
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[0]
    env.saveFile = False
    env.reset()
    env.first_time = False
    Question = 1
    Q, number_of_steps, total_return, total_return_steps = SMDPq_learning(env, num_episodes=num_episodes,iterations=iterations, gamma=0.9  , alpha=1/8, epsilon=0.2)
    
    plt.loglog(range(num_episodes),number_of_steps)
    # env.my_state = env.start_state
    if Question != 1:
        env.start_state = 90
    # V = np.zeros(104)
    # optimal_policys = np.zeros(104)
    # optimal_policys += -1
    # for k, v in Q.items():
    #     if k < 104:
    #         if not np.allclose(v,v[0]):
    #             optimal_policys[k] = np.argmax(v)
    #         V[k] = np.max(v)
    # V = preprocessing.minmax_scale(V, feature_range=(0, 0.5))
    # env.V = V
    # env.optimal_policy = optimal_policys
    # env.draw_circles = True
    # env.draw_arrows = False
    # env.figtitle = figtitle[pb]+'_'+str(num_episodes)
    # env.render(mode='human')
    # env.draw_circles = False
    # env.draw_arrows = True
    # env.render(mode='human')
print("done option action  learning")
for pb in range(1):
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[0]
    env.saveFile = False
    env.n_actions = 4
    env.reset()
    env.first_time = False
    Question = 1
    Q, number_of_steps, total_return, total_return_steps = q_learning(env, num_episodes=num_episodes,iterations=iterations, gamma=0.9, alpha=1/8, epsilon=0.2)
    
    plt.loglog(range(num_episodes),number_of_steps)
    #plt.show()
    # plt.plot(total_return)
    # plt.show()
    # if Question != 1:
    #     env.state = 90
    #     env.start_state = 90
    # V = np.zeros(104)
    # optimal_policys = np.zeros(104)
    # optimal_policys += -1
    # for k, v in Q.items():
    #     if k < 104:
    #         if not np.allclose(v,v[0]):
    #             optimal_policys[k] = np.argmax(v)
    #         V[k] = np.max(v)
    # V = preprocessing.minmax_scale(V, feature_range=(0, 0.5))
    # env.V = V
    # env.optimal_policy = optimal_policys
    # env.draw_circles = True
    # env.draw_arrows = False
    # env.figtitle = figtitle[pb]+'_'+str(num_episodes)
    # env.render(mode='human')
    # env.draw_circles = False
    # env.draw_arrows = True
    # env.render(mode='human')
print("done action leanrnig")
for pb in range(1):
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[0]
    env.saveFile = False
    env.reset()
    env.first_time = False
    Question = 1
    #env.n_actions = 
    Q, number_of_steps, total_return, total_return_steps = Intra_Option_q_learning(env, num_episodes=num_episodes,iterations=iterations, gamma=0.9, alpha= 1, epsilon=0.2)
    
    plt.loglog(range(num_episodes),number_of_steps)
    # if Question != 1:
    #     env.state = 90
    #     env.start_state = 90
    # V = np.zeros(104)
    # optimal_policys = np.zeros(104)
    # optimal_policys += -1
    # for k, v in Q.items():
    #     if k < 104:
    #         if not np.allclose(v,v[0]):
    #             optimal_policys[k] = np.argmax(v)
    #         V[k] = np.max(v)
    # V = preprocessing.minmax_scale(V, feature_range=(0, 0.5))
    # env.V = V
    # env.optimal_policy = optimal_policys
    # env.draw_circles = True
    # env.draw_arrows = False
    # env.figtitle = figtitle[pb]+'_'+str(num_episodes)
    # env.render(mode='human')
    # env.draw_circles = False
    # env.draw_arrows = True
    # env.render(mode='human')
plt.xlabel("Episodes")
plt.ylabel("Steps Taken to Reach goal")
plt.title("Reaching Goal G1")
plt.legend(("SMDP_Qlearning","Actions Only","Intra Option"))
plt.savefig("Reaching_GoalIntra_G1Q1")
plt.show()
print("done")
plt.clf()
for pb in range(1):
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[0]
    env.saveFile = False
    env.reset()
    env.first_time = False
    Question = 0
    Q, number_of_steps, total_return, total_return_steps = SMDPq_learning(env, num_episodes=num_episodes,iterations=iterations, gamma=0.9, alpha=1/8, epsilon=0.1)
    
    plt.loglog(range(num_episodes),number_of_steps)
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[0]
    env.saveFile = False
    env.n_actions = 4
    env.reset()
    env.first_time = False
    Question = 0
    Q, number_of_steps, total_return, total_return_steps = q_learning(env, num_episodes=num_episodes,iterations=iterations, gamma=0.9, alpha=1/8, epsilon=0.1)
    
    plt.loglog(range(num_episodes),number_of_steps)
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[0]
    env.saveFile = False
    env.reset()
    env.first_time = False
    Question = 0
    # env.n_actions = 2
    Q, number_of_steps, total_return, total_return_steps = Intra_Option_q_learning(env, num_episodes=num_episodes,iterations=iterations, gamma=0.9, alpha= 1, epsilon=0.1)
    
    plt.loglog(range(num_episodes),number_of_steps)
plt.xlabel("Episodes")
plt.ylabel("Steps Taken to Reach goal")
plt.title("Reaching Goal G1")
plt.legend(("SMDP_Qlearning","Actions Only","Intra Option"))
plt.savefig("Raeaching_GoalNTra_G1Q2")
print("done")
plt.clf()
for pb in range(1):
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[1]
    env.saveFile = False
    env.reset()
    env.first_time = False
    Question = 0
    Q, number_of_steps, total_return, total_return_steps = SMDPq_learning(env, num_episodes=num_episodes,iterations=iterations, gamma=0.9, alpha=1/8, epsilon=0.1)
    
    plt.loglog(range(num_episodes),number_of_steps)
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[1]
    env.saveFile = False
    env.n_actions = 4
    env.reset()
    env.first_time = False
    Question = 0
    Q, number_of_steps, total_return, total_return_steps = q_learning(env, num_episodes=num_episodes,iterations=iterations, gamma=0.9, alpha=1/8, epsilon=0.1)
    
    plt.loglog(range(num_episodes),number_of_steps)
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[1]
    env.saveFile = False
    env.reset()
    env.first_time = False
    Question = 0
    # env.n_actions = 2
    Q, number_of_steps, total_return, total_return_steps = Intra_Option_q_learning(env, num_episodes=num_episodes,iterations=iterations, gamma=0.9, alpha= 1, epsilon=0.1)
    
    plt.loglog(range(num_episodes),number_of_steps)

plt.xlabel("Episodes")
plt.ylabel("Steps Taken to Reach goal")
plt.title("Reaching Goal G2")
plt.legend(("SMDP_Qlearning","Actions Only","Intra Option"))
plt.savefig("Raeaching_Goal__INTRA_G2Q2")
plt.clf()
for pb in range(1):
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[1]
    env.saveFile = False
    env.reset()
    env.first_time = False
    Question = 1
    Q, number_of_steps, total_return, total_return_steps = SMDPq_learning(env, num_episodes=num_episodes,iterations=iterations, gamma=0.9, alpha=1/8, epsilon=0.1)
    
    plt.loglog(range(num_episodes),number_of_steps)
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[1]
    env.saveFile = False
    env.n_actions = 4
    env.reset()
    env.first_time = False
    Question = 1
    Q, number_of_steps, total_return, total_return_steps = q_learning(env, num_episodes=num_episodes,iterations=iterations, gamma=0.9, alpha=1/8, epsilon=0.1)
    
    plt.loglog(range(num_episodes),number_of_steps)
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[1]
    env.saveFile = False
    env.reset()
    env.first_time = False
    Question = 1
    # env.n_actions = 2
    Q, number_of_steps, total_return, total_return_steps = Intra_Option_q_learning(env, num_episodes=num_episodes,iterations=iterations, gamma=0.9, alpha= 1, epsilon=0.1)
    
    plt.loglog(range(num_episodes),number_of_steps)
plt.xlabel("Episodes")
plt.ylabel("Steps Taken to Reach goal")
plt.title("Reaching Goal G2")
plt.legend(("SMDP_Qlearning","Actions Only","Intra Option"))
plt.savefig("Raeaching_Goal_Inta_G2Q1")