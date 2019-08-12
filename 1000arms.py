import numpy as np
from pylab import *
import tqdm
import math




act_mean = np.random.normal(0,1,(1000,2000))


def epsilon_greedyv(eps,means):
    action = np.argmax(means,axis = 0).reshape(2000,1)
    nongreedy_action = np.random.choice(range(1000),(2000,1))
    prob = np.random.rand(2000,1)
    kk = np.where(prob<eps)[0]
    action[kk] = nongreedy_action[kk]; 
    return action;




av_greedy = np.zeros((1000,1))
av_greedy2 = np.zeros((1000,1))
av_greedy3 = np.zeros((1000,1))
av_greedyp = np.zeros((1000,1))
av_greedyp2 = np.zeros((1000,1))
av_greedyp3 = np.zeros((1000,1))
est_mean = np.zeros((1000,2000))
est_mean2 = np.zeros((1000,2000))
est_mean3 = np.zeros((1000,2000))
count = np.zeros((1000,2000))
count2 = np.zeros((1000,2000))
count3 = np.zeros((1000,2000))
for i in tqdm.tqdm(range(1000 )):
        action = np.argmax(est_mean,axis = 0);
        action2 = epsilon_greedyv(0.1,est_mean2);
        #action3 = epsilon_greedyv(0.01,est_mean3);
        reward = np.zeros(2000)
        reward2 = np.zeros(2000)
        #reward3 = np.zeros(2000)
        actionp = 0
        actionp2 = 0
        for r in range(2000):
            reward[r] = np.random.normal(act_mean[action[r],r],1)
            reward2[r] = np.random.normal(act_mean[action2[r],r],1)
            #reward3[r] = np.random.normal(act_mean[action3[r],r],1)
            count[action[r],r] = count[action[r],r] + 1;
            count2[action2[r],r] = count2[action2[r],r] + 1
            #count3[action3[r],r] = count3[action3[r],r] + 1
            est_mean[action[r],r] = ((count[action[r],r] - 1)*est_mean[action[r],r] + reward[r])/(count[action[r],r])
            est_mean2[action2[r],r] = ((count2[action2[r],r] - 1)*est_mean2[action2[r],r] + reward2[r])/(count2[action2[r],r])
            highest_arm = np.max(act_mean[:,r])
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
        actionp3 = 0
        for r in range(2000):
            reward3[r] = np.random.normal(act_mean[action3[r],r],1)
            count3[action3[r],r] = count3[action3[r],r] + 1
            est_mean3[action3[r],r] = ((count3[action3[r],r] - 1)*est_mean3[action3[r],r] + reward3[r])/(count3[action3[r],r])
            highest_arm = np.max(act_mean[:,r])
            actionp3 = actionp3 + (act_mean[action3[r],r]/highest_arm)*100
        av_greedy3[i] = np.sum(reward3)
        av_greedyp3[i] = av_greedyp3[i] + actionp3/2000


UCB = np.zeros((1000,1))
UCBp = np.zeros((1000,1))
c = 5;
for i in tqdm.tqdm(range(2000)):
    est_meanu = np.zeros((1000,1));
    upperbound=np.ones((1000,1))
    countu = np.zeros((1000,1))
    for k in range(1000):
        actionu = np.argmax(est_meanu + upperbound)
        rewardu = np.random.normal(act_mean[actionu,i],1)
        countu[actionu] = countu[actionu] + 1;
        est_meanu[actionu] = ((countu[actionu] - 1)*est_meanu[actionu] + rewardu)/(countu[actionu] + 1)
        if(countu[actionu] != 1):
            upperbound[actionu] = c*sqrt(abs(math.log(k+1))/(countu[actionu] - 1))
        UCB[k] = UCB[k] + rewardu;
        highest_arm = np.max(act_mean[:,i])
        UCBp[k] = UCBp[k] + (act_mean[actionu,i]/highest_arm)*100








UCB1 = np.zeros((1000,1))
UCBp1 = np.zeros((1000,1))
c = 2;
for i in tqdm.tqdm(range(2000)):
    est_meanu = np.zeros((1000,1));
    upperbound=np.ones((1000,1))
    countu = np.zeros((1000,1))
    for k in range(1000):
        actionu = np.argmax(est_meanu + upperbound)
        rewardu = np.random.normal(act_mean[actionu,i],1)
        countu[actionu] = countu[actionu] + 1;
        est_meanu[actionu] = ((countu[actionu] - 1)*est_meanu[actionu] + rewardu)/(countu[actionu] + 1)
        if(countu[actionu] != 1):
            upperbound[actionu] = c*sqrt(abs(math.log(k+1))/(countu[actionu] - 1))
        UCB1[k] = UCB1[k] + rewardu;
        highest_arm = np.max(act_mean[:,i])
        UCBp1[k] = UCBp1[k] + (act_mean[actionu,i]/highest_arm)*100





UCB2 = np.zeros((1000,1))
UCBp2 = np.zeros((1000,1))
c = 1;
for i in tqdm.tqdm(range(2000)):
    est_meanu = np.zeros((1000,1));
    upperbound=np.ones((1000,1))
    countu = np.zeros((1000,1))
    for k in range(1000):
        actionu = np.argmax(est_meanu + upperbound)
        rewardu = np.random.normal(act_mean[actionu,i],1)
        countu[actionu] = countu[actionu] + 1;
        est_meanu[actionu] = ((countu[actionu] - 1)*est_meanu[actionu] + rewardu)/(countu[actionu] + 1)
        if(countu[actionu] != 1):
            upperbound[actionu] = c*sqrt(abs(math.log(k+1))/(countu[actionu] - 1))
        UCB2[k] = UCB2[k] + rewardu;
        highest_arm = np.max(act_mean[:,i])
        UCBp2[k] = UCBp2[k] + (act_mean[actionu,i]/highest_arm)*100



plot(range(1000),UCB[:1000]/2000)
plot(range(1000),UCB1[:1000]/2000)
plot(range(1000),UCB2[:1000]/2000)
legend(("C = 5","C = 2","C = 1"))
xlabel("Steps");ylabel("Average reward")
title("UCB Average Reward")
savefig("Answer1000avg1")
show()
plot(range(1000),av_greedy[:1000]/2000)
plot(range(1000),av_greedy2[:1000]/2000)
plot(range(1000),av_greedy3[:1000]/2000)
legend(("Eps = 0","Eps = 0.1","Eps = 0.01"))
xlabel("Steps");ylabel("Average reward")
title("Epsilon-Greedy Average Reward")
savefig("Answer1000avg")
show()


plot(range(1000),UCBp[:1000]/2000)
plot(range(1000),UCBp1[:1000]/2000)
plot(range(1000),UCBp2[:1000]/2000)
legend(("C = 5","C = 2","C = 1"))
xlabel("Steps");ylabel("Percentage Optimal Action ")
title("UCB Percentage Optimal Action")
savefig("Answer10002.png")
show()

plot(range(1000),av_greedyp[:1000])
plot(range(1000),av_greedyp2[:1000])
plot(range(1000),av_greedyp3[:1000])
legend(("Eps = 0","Eps = 0.1","Eps = 0.01"))
xlabel("Steps");ylabel("Percentage Optimal Action ")
title("Epsilon-Greedy Optimal Action")
savefig("Answer10001.png")
show()




plot(range(1000),av_greedy[:1000]/2000)
plot(range(1000),av_greedy2[:1000]/2000)
plot(range(1000),av_greedy3[:1000]/2000)
plot(range(1000),UCB1[:1000]/2000)
plot(range(1000),UCB2[:1000]/2000)
plot(range(1000),UCB[:1000]/2000)
legend(("greedy", "eps = 0.1","eps= 0.01","UCB c =2","UCB c = 1","UCB c = 5" ))
xlabel("Steps");ylabel("Average Reward")
title("Comparision for 1000 arm bandits")
savefig("Answer4")
show()


plot(range(1000),av_greedyp[:1000])
plot(range(1000),av_greedyp2[:1000])
plot(range(1000),av_greedyp3[:1000])
plot(range(1000),UCBp1[:1000]/2000)
plot(range(1000),UCBp2[:1000]/2000)
plot(range(1000),UCBp[:1000]/2000)
legend(("greedy", "eps = 0.1","eps= 0.01","UCB c =2","UCB c = 1","UCB c = 5" ))
xlabel("Steps");ylabel("Average Reward")
title("Comparision for 1000 arm bandits")
savefig("Answer4op")
show()