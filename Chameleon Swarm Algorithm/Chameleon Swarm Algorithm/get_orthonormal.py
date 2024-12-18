import numpy as np


def get_orthonormal(m , n):
    """产生一组M*N 的正交向量

    Args:
        m (int): 
        n (int): 要求n小于等于m

    Returns:
        array: 
    """
    count = 0
    while(count==0):
        A = np.random.rand(m, m)
        B = A.T * A

        D,P = np.linalg.eig(B)

        if(((P.T*P)-np.eye(m)> np.spacing(1)).all()):
            count = 0
        else:
            answer = P[:,:n]
            count = 1
    return answer


# print(get_orthonormal(5, 4))