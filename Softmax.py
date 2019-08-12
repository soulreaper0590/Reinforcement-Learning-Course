import numpy as np
from pylab import *
import tqdm



act_mean = np.random.normal(0,1,(10,2000))


soft_max1 = np.zeros((1000,1))
soft_max1p = np.zeros((1000,1))
temp = 0.1
for k in tqdm.tqdm(range(2000)):
    est_means = np.zeros((10,1))
    counts = np.zeros((10,1))
    for i in range(1000):
        exp_means = exp(est_means/temp);
        sum = np.sum(exp_means)
        exp_means = exp_means/sum;
        choice = np.random.rand()
        for r in range(10):
            if(np.sum(exp_means[:r+1])>choice):
                actions = r
                break;
        counts[actions] = counts[actions] + 1;
        reward = np.random.normal(act_mean[actions,k],1)
        est_means[actions] = ((counts[actions] - 1)*est_means[actions] + reward)/(counts[actions] + 1)
        soft_max1[i] = reward + soft_max1[i];
        highest_arm = np.max(act_mean[:,k])
        soft_max1p[i] = soft_max1p[i] + (act_mean[actions,k]/highest_arm)*100


soft_max2 = np.zeros((1000,1))
soft_max2p = np.zeros((1000,1))
temp = 0.01
for k in tqdm.tqdm(range(2000)):
    est_means = np.zeros((10,1))
    counts = np.zeros((10,1))
    for i in range(1000):
        exp_means = exp(est_means/temp);
        sum = np.sum(exp_means)
        exp_means = exp_means/sum;
        choice = np.random.rand()
        for r in range(10):
            if(np.sum(exp_means[:r+1])>choice):
                actions = r
                break;
        counts[actions] = counts[actions] + 1;
        reward = np.random.normal(act_mean[actions,k],1)
        est_means[actions] = ((counts[actions] - 1)*est_means[actions] + reward)/(counts[actions] + 1)
        soft_max2[i] = reward + soft_max2[i];
        highest_arm = np.max(act_mean[:,k])
        soft_max2p[i] = soft_max2p[i] + (act_mean[actions,k]/highest_arm)*100


soft_max3 = np.zeros((1000,1))
soft_max3p = np.zeros((1000,1))
temp = 1
for k in tqdm.tqdm(range(2000)):
    est_means = np.zeros((10,1))
    counts = np.zeros((10,1))
    for i in range(1000):
        exp_means = exp(est_means/temp);
        sum = np.sum(exp_means)
        exp_means = exp_means/sum;
        choice = np.random.rand()
        for r in range(10):
            if(np.sum(exp_means[:r+1])>choice):
                actions = r
                break;
        counts[actions] = counts[actions] + 1;
        reward = np.random.normal(act_mean[actions,k],1)
        est_means[actions] = ((counts[actions] - 1)*est_means[actions] + reward)/(counts[actions] + 1)
        soft_max3[i] = reward + soft_max3[i];
        highest_arm = np.max(act_mean[:,k])
        soft_max3p[i] = soft_max3p[i] + (act_mean[actions,k]/highest_arm)*100



plot(range(1000),soft_max1[:1000]/2000)
plot(range(1000),soft_max2[:1000]/2000)
plot(range(1000),soft_max3[:1000]/2000)
legend(("Temp = 0.1","Temp = 0.01","Temp = 1"))
xlabel("Steps");ylabel("Average reward")
title("Average Reward Softmax")
savefig("Answer2")
show()



plot(range(1000),soft_max1p/2000)
plot(range(1000),soft_max2p/2000)
plot(range(1000),soft_max3p/2000)
legend(("Temp = 0.1","Temp = 0.01","Temp = 1"))
xlabel("Steps");ylabel("Optimal Action (%)")
title("Optimal Action Softmax")
savefig("Answer2op")
show()