import numpy as np
from pylab import *
import tqdm




def epsilon_greedyv(eps,means):
    action = np.argmax(means,axis = 0).reshape(2000,1)
    nongreedy_action = np.random.choice(range(10),(2000,1))
    prob = np.random.rand(2000,1)
    kk = np.where(prob<eps)[0]
    action[kk] = nongreedy_action[kk]; 
    return action;




act_mean = np.random.normal(0,1,(10,2000))





av_greedy = np.zeros((1000,1))
av_greedy2 = np.zeros((1000,1))
av_greedy3 = np.zeros((1000,1))
av_greedyp = np.zeros((1000,1))
av_greedyp2 = np.zeros((1000,1))
av_greedyp3 = np.zeros((1000,1))
est_mean = np.zeros((10,2000))
est_mean2 = np.zeros((10,2000))
est_mean3 = np.zeros((10,2000))
count = np.zeros((10,2000))
count2 = np.zeros((10,2000))
count3 = np.zeros((10,2000))
for i in tqdm.tqdm(range(1000 )):
        action = np.argmax(est_mean,axis = 0);
        action2 = epsilon_greedyv(0.1,est_mean2);
        #action3 = epsilon_greedyv(0.01,est_mean3);
        reward = np.zeros(2000)
        reward2 = np.zeros(2000)
        actionp = 0
        actionp2 = 0
        #reward3 = np.zeros(2000)
        for r in range(2000):
            reward[r] = np.random.normal(act_mean[action[r],r],1)
            reward2[r] = np.random.normal(act_mean[action2[r],r],1)
            highest_arm = np.max(act_mean[:,r])
            #reward3[r] = np.random.normal(act_mean[action3[r],r],1)
            count[action[r],r] = count[action[r],r] + 1;
            count2[action2[r],r] = count2[action2[r],r] + 1
            #count3[action3[r],r] = count3[action3[r],r] + 1
            est_mean[action[r],r] = ((count[action[r],r] - 1)*est_mean[action[r],r] + reward[r])/(count[action[r],r])
            est_mean2[action2[r],r] = ((count2[action2[r],r] - 1)*est_mean2[action2[r],r] + reward2[r])/(count2[action2[r],r])
            actionp = actionp + (act_mean[action[r],r]/highest_arm)*100;
            actionp2 = actionp2 + (act_mean[action2[r],r]/highest_arm)*100;
            #est_mean3[action3[r],r] = ((count3[action3[r],r] - 1)*est_mean3[action3[r],r] + reward3[r])/(count3[action3[r],r])
        av_greedy[i] = np.sum(reward)
        av_greedy2[i] = np.sum(reward2)
        av_greedyp[i] = av_greedyp[i] + actionp/2000
        av_greedyp2[i] = av_greedyp2[i] + actionp2/2000




for i in tqdm.tqdm(range(1000 )):
        action3 = epsilon_greedyv(0.01,est_mean3);
        reward3 = np.zeros(2000)
        actionp3 = 0;
        for r in range(2000):
            reward3[r] = np.random.normal(act_mean[action3[r],r],1)
            count3[action3[r],r] = count3[action3[r],r] + 1
            highest_arm = np.max(act_mean[:,r]);
            actionp3 = actionp3 + (act_mean[action3[r],r]/highest_arm)*100
            est_mean3[action3[r],r] = ((count3[action3[r],r] - 1)*est_mean3[action3[r],r] + reward3[r])/(count3[action3[r],r])
        av_greedy3[i] = np.sum(reward3)
        av_greedyp3[i] = av_greedyp3[i] + actionp3/2000




plot(range(1000),av_greedy[:1000]/2000)
plot(range(1000),av_greedy2[:1000]/2000)
plot(range(1000),av_greedy3[:1000]/2000)
#plot(range(1000),soft_max1[:1000]/2000)
legend(("greedy", "eps = 0.1","eps= 0.01"))
xlabel("Steps");ylabel("Average Reward")
title("Average Reward EPS-Greedy")
savefig("Answer1")
show()




plot(range(1000),av_greedyp[:1000])
plot(range(1000),av_greedyp2[:1000])
plot(range(1000),av_greedyp3[:1000])
#plot(range(1000),soft_max1[:1000]/2000)
legend(("greedy", "eps = 0.1","eps= 0.01"))
xlabel("Steps");ylabel("Optimal Action(%)")
title("Optimal EPS-Greedy")
savefig("Answer1op")
show()