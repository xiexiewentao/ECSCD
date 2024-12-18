import numpy as np

from Chameleon import Chameleon

dim = 2;
ub = 50 * np.ones((1, 2))
lb = -50 * np.ones((1, 2))
noP = 30;
maxIter = 1000;

bestFitness, bestPostion, CSAConvcurve = Chameleon(noP, maxIter, lb, ub, dim)

print("The optimal fitness value found by Standard Chameleon is")
print(bestFitness)
