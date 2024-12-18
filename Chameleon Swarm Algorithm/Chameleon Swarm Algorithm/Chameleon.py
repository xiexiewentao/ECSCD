from math import exp, log, sqrt
import numpy as np
from Objfun import Objfun

from initialization import initialization
def Chameleon(searchAgents, iteMax, lb, ub, dim):
    """_summary_

    Args:
        searchAgents (_type_): _description_
        iteMax (_type_): _description_
        lb (_type_): _description_
        ub (_type_): _description_
        dim (_type_): _description_
        
    """

    if ub.shape[1] == 1:
        ub = np.ones(1, dim)*ub
        lb = np.ones(1, dim)*lb

    cg_curve = np.zeros((1, iteMax))

    #初始化种群
    chameleonPositions=initialization(searchAgents,dim,ub,lb);

    #评估初始种群的适应度
    fit = np.zeros((searchAgents, 1));

    for i in range(0, searchAgents):
       fit[i, 0] = Objfun(chameleonPositions[i,:]) 

    #初始化CSA的参数
    fitness = fit
    fmin0 = np.min(fit, axis=0)
    index = np.argmin(fit, axis=0)
    chameleonBestPosition = chameleonPositions
    gPosition = chameleonPositions[index,:].reshape(1, 2)
    velocity = 0.1 * chameleonBestPosition
    v0 = 0.0*velocity

    rho = 1.0
    p1 = 2.0
    p2 = 2.0
    c1 = 2.0
    c2 = 1.8
    gamma = 2.0
    alpha = 4.0
    beta = 3.0

    #Start CSA
    for t in range(1, iteMax+1):
        a = 2590*(1-exp(-log(t)))
        omega = (1-(t/iteMax))**(rho*sqrt(t/iteMax))
        p1 = 2*exp(-2*(t/iteMax)**2)
        p2 = 2/(1+exp((-t+iteMax/2)/100))
        mu = gamma*exp(-(alpha*t/iteMax)**beta)
        ch = np.ceil(searchAgents*np.random.rand(1, searchAgents)).astype(int) - 1

    #Update the position of CSA
        for i in range(0, searchAgents):
            if np.random.random_sample()>0.1:
                chameleonPositions[i,:] = chameleonPositions[i,:] + p1*(chameleonBestPosition[ch[0,i],:]-chameleonPositions[i,:])*np.random.random_sample()+p2*(gPosition-chameleonPositions[i,:])*np.random.random_sample()
            else:
                for j in range(0, dim):
                    chameleonPositions[i,j] = gPosition[0][j]+mu*(ub[0][j]- lb[0][j]*np.sign(np.random.random_sample()-0.50))
        

        #Chameleon velocity updates and find a food source
        '''
        当第一次迭代时,t = 1, a = 0;在下面公式中会发生除0错,但在源代码中未作出处理，此处对a加了一个eps
        '''
        for i in range(0, searchAgents):
            velocity[i,:] = omega*velocity[i,:] + p1*(chameleonBestPosition[i,:]-chameleonPositions[i,:])*np.random.random_sample()+p2*(gPosition-chameleonPositions[i,:])*np.random.random_sample()
            chameleonPositions[i,:] = chameleonPositions[i,:]+ (np.power(velocity[i,:],2) - np.power(v0[i,:],2))/(2*(a+np.spacing(1)))
        v0 = velocity;


        #handling boundary violations
        for i in range(0, searchAgents):
            if (chameleonPositions[i,:] < lb).all():
                chameleonPositions[i,:] = lb
            elif(chameleonPositions[i,:] > ub).all():
                chameleonPositions[i,:] = ub


        #Relocation of chameleon positions
        for i in range(0, searchAgents):
            ub_ = np.sign(chameleonPositions[i,:]-ub)>0
            lb_ = np.sign(chameleonPositions[i,:]-lb)<0

            chameleonPositions[i,:] = (chameleonPositions[i,:]*(np.logical_not(np.logical_xor(lb_,ub_)))) + ub*ub_ + lb*lb_
            fit[i,0] = Objfun(chameleonPositions[i,:])
            if (fit[i]<fitness[i]).all():
                chameleonBestPosition[i,:] = chameleonPositions[i,:]
                fitness[i] = fit[i]

        #Evaluate the new positions
        fmin = np.min(fitness, axis=0)
        index = np.argmin(fitness, axis=0)
        if (fmin < fmin0).all():
            gPosition = chameleonBestPosition[index,:]
            fmin0 = fmin


        cg_curve[0,t-1] = fmin0
    
    ngPosition = np.nonzero(fitness == np.min(fitness,axis=0))
    g_best = chameleonBestPosition[ngPosition[0],:]
    fmin0 = Objfun(g_best)
    
    return fmin0, gPosition, cg_curve