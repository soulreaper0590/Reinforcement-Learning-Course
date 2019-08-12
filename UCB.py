import numpy as np
from pylab import *
import tqdm
import math


act_mean = np.random.normal(0,1,(10,2000))

UCB = np.zeros((1000,1))
UCBp = np.zeros((1000,1))
c = 5;
for i in tqdm.tqdm(range(2000)):
    est_meanu = np.zeros((10,1));
    upperbound=np.ones((10,1))
    countu = np.zeros((10,1))
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
    est_meanu = np.zeros((10,1));
    upperbound=np.ones((10,1))
    countu = np.zeros((10,1))
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
    est_meanu = np.zeros((10,1));
    upperbound=np.ones((10,1))
    countu = np.zeros((10,1))
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
plt.title("UCB Average Reward")
savefig("Answer3")
show()



plot(range(1000),UCBp[:1000]/2000)
plot(range(1000),UCBp1[:1000]/2000)
plot(range(1000),UCBp2[:1000]/2000)
legend(("C = 5","C = 2","C = 1"))
xlabel("Steps");ylabel("Percentage Optimal Action ")
title("UCB Percentage Optimal Action")
savefig("Answer3op =")
show()