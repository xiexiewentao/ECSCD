import numpy as np


def initialization(searchAgent, dim, u, l):
    """初始化第一批变色龙

    Args:
        searchAgent (int): 代理的数量
        dim (int): 问题的维度
        u (array): 搜索区域的上界
        l (array): 搜索区域的下界

    Returns:
        _type_: _description_ 
    """

    """ 初始化返回值，
        由于在matlab源代码在循环中对pos采用列的方式,
        在numpy中对列操作一直有错误,
        故我采用行的方式,
        在赋值完成后进行转置,
        所以此处初始化返回值行列相反
    """
    pos = np.zeros((dim, searchAgent)) 

    Boundary_no = u.shape[1]
    if Boundary_no == 1:
        u_new = np.ones((1, dim)) * u
        l_new = np.ones((1, dim)) * l
    else:
        u_new = u
        l_new = l

    for i in range(0, dim):
        u_i = u_new[:, i]
        l_i = l_new[:, i]
        pos[i,:] = np.random.rand(1, searchAgent)*(u_i - l_i) + l_i
    pos = pos.T

    return pos


# u = np.ones((1, 2)) * 50
# l = np.ones((1, 2)) * -50
# dim = 2
# searchAgent = 30

# print(initialization(searchAgent, dim, u, l).shape)