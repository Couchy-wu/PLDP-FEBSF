"""
This function is mainly used to implement the overall flow of the EM algorithm.
"""
import numpy as np
import random as rd
from time import sleep
from scipy.optimize import minimize, LinearConstraint
from scipy.optimize import Bounds
import sys
import os


def getprob_x_z(q, epsilon, x_i, domain):
    """
    The main role of this function is to specify the probability of observing the data as x_i after the privacy budget and original data distribution.
    q:Raw distribution of data
    epsilon:Designated privacy budget
    x_i:Observed data
    domain:Value range of the data
    """
    probx = 0.0  # initialization
    K = len(q)
    for i in range(K):
        probx += q[i] * getconditionprob(x_i, domain[i], epsilon, K)
        
    return probx  



def getprobiltyz(ep, q, epsilon_set, ei, x_i, domain):
    """
   This function is mainly used to calculate the probability of perturbing a given observation x_i using the privacy budget ei.
    """
    assert len(ep) == len(epsilon_set)  # Determine if the lengths are consistent
    molecular = ep[ei] * getprob_x_z(q, epsilon_set[ei], x_i, domain)
    T = len(epsilon_set)  # Number of privacy budgets
    denominator = 0.0
    for t in range(T):
        denominator += ep[t] * getprob_x_z(q, epsilon_set[t], x_i, domain)
    probz = molecular / denominator
    return probz


def getconditionprob(x1, x2, epsilon, k):
    """
    The role of this function is to calculate p(x_i|x_j,epsilon) given the values x_i,x_j and the privacy budget epsilon.
    """
    if x1 == x2:
        return np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    else:
        return 1 / (np.exp(epsilon) + k - 1)  # Returns the final conditional probability.


def updateq_j(perbdata, q, ep, domain, epsilon_set, j):
    """
    This function serves to update the value of the parameter q_j.
    """
    p_j = 0.0  #  Initialization parameters.
    N = len(perbdata)  # Number of observations.
    T = len(epsilon_set)  # Number of privacy budgets.
    for i in range(N):
        pt=0.0
        for t in range(T):
            pt+= getconditionprob(
                perbdata[i], domain[j], epsilon_set[t],len(domain)
            ) * getprobiltyz(ep, q, epsilon_set, t, perbdata[i], domain)
        # print(f"pt:{pt}")
        p_j+=pt
    
    p_j=p_j /N
    # print(f"p_{j}:{p_j}:")
    return p_j # Returns the updated parameters.

def getsump(p,x_i,epsilon,domain):
    """
    The main role of this function is to realize the computation of probability summation given a privacy budget.
    """
    assert(len(p)==len(domain))
    K=len(domain)
    result=0.0
    # print(f"p:{p}")
    for k in range(K):
        result+=p[k]*getconditionprob(x_i,domain[k],epsilon,len(domain))
    if result<=0:
        print(f"p:{p},domain:{domain}")
    return result #  Return results.



def objfun(p,ep,perbdatalist,epsilon_set,domain,L2):
    """
    This function is the objective function that needs to be optimized eventually.
    """
    assert(len(p)==len(domain)) 
    N=len(perbdatalist)
    T=len(epsilon_set) 
    K=len(domain) 
    result=0.0 #The final experimental results returned.

    for i in range(N):
        # print(f"i{i}")
        for z in range(T):
            # print(f"ep[z]{ep[z]}")
            assert(ep[z]>=0) #Let's first determine if the constraints are met.
            assert(getsump(p,perbdatalist[i],epsilon_set[z],domain)>0) 
            result+=(np.log(ep[z])+np.log(getsump(p,perbdatalist[i],epsilon_set[z],domain))) 
    
    result+=L2*(sum(ep)-1) 
    return -result 




def constrain(p):
    """
    This function is a constraint that needs to be satisfied for the parameter.
    """
    return sum(p)-1 # probability sum to one (math.)




def updatp(perbdata, q, ep, domian, epsilon_set):
    """
    The role of this function is mainly to update the distribution probability q of the original data
    """
    N = len(perbdata)  
    T = len(epsilon_set)  
    K = len(domian)  
    newq = [0] * K
    for j in range(K):
        newq[j]=updateq_j(perbdata,q,ep,domian,epsilon_set,j)
    return newq


def updatep(perbdata, q, ep, epsilon_set, domain):
    """
    The main purpose of this function is to update the distribution of users across privacy budgets ep.
    """
    t = len(epsilon_set) 
    newep = [0] * t
    N = len(perbdata)  
    for i in range(t):
        sum1 = 0.0
        for j in range(N):
            sum1 += getprobiltyz(ep, q, epsilon_set, i, perbdata[j], domain)
        newep[i] = sum1 / N

    return newep  # Bringing the new updated values into the carry distribution.


def computgap(ep1, q1, ep2, q2):
    """
    This function is mainly used to calculate the L2 interval between two functions.
    """
    gap = 0.0
    assert (len(ep1) + len(q1)) == (len(ep2) + len(q2))
    n1 = len(ep1)
    n2 = len(q1)
    for i in range(n1):
        gap += (ep1[i] - ep2[i]) ** 2
    for i in range(n2):
        gap += (q1[i] - q2[i]) ** 2
    return gap  # Returns the interval between two vectors.


def train(perdata, ep, q, epsilon_set, domain):
    """
   The main function of this function is to iterate the EM algorithm 
   given the perturbation data, the privacy set and the initial values of each hidden variable.
    """
    gap = 10**-7  # Termination conditions of training.
    maxsteps = 2000  # Maximum number of iterations.
    newep = []
    newq = []
    step = 1
    while step < maxsteps:
        print(f"step:{step},ep:{ep},q:{q}")  # Printing Iteration Results.
        newep = updatep(perdata, q, ep, epsilon_set, domain)  
        print(f"newep:{newep}")
        newq = updatp(perdata, q, ep, domain, epsilon_set)
        print(f"newq:{newq}")
        # if computgap(ep, q, newep, newq) < gap:  
        #     break
        ep[:] = newep[:]
        q[:] = newq[:]
        step = step + 1  # Perform the next iteration.
    return newq  # Returns the value of the EM algorithm after iteration.


cons = { "type": "eq", "fun":constrain} # Define the constraints of the optimization function

def predealq(q):
    """
    The main purpose of this function is to preprocess the initialization 
    parameters, the initial values output in the basic model, in the presence of negative numbers.
    """
    n=len(q)
    newq=[0]*n
    sum=0.0
    for i in range(n):
        if q[i]<0:
           q[i]=10**-6
        newq[i]=q[i]
        sum+=newq[i]
    newq=[v/sum for v in newq]
    return newq # Returns the preprocessed value

def preinitial(len_ep,len_q):
    """
    The main purpose of this function is a method to 
    initialize the EM algorithm in case the client does not send a perturbed privacy budget.
    """
    ep=[1/len_ep]*len_ep
    q=[1/len_q]*len_q

    return ep,q



def newtrain(perdatalist,ep,q,epsilon_set,domain):
    """
    This training model solves the parameters in the EM model by solving the 
    numerical solution, not by analytical solution.
    """
    L=len(q) 
    q=predealq(q)
    ep=predealq(ep)
    # ep,q=preinitial(len(ep),len(q)) 
    # ep=Getrandprob(len(ep))
    # q=Getrandprob(len(q))
    # ep=predealq(ep)
    bound=Bounds([1e-6]*L, [1]*L)

    gap=10**(-10) 
    maxsteps=5 
    newep=[]
    newp=[] 
    step=1 
    L2=-len(perdatalist) 
    finalfun=0.0
    while step < maxsteps:
        print(f"step:{step},ep:{ep},q:{q}")  
        newep = updatep(perdatalist, q, ep, epsilon_set, domain)  
        tempresult=minimize(objfun,q,args=(ep,perdatalist,epsilon_set,domain,L2),method='SLSQP',constraints=cons,bounds=bound) #优化函数
        newp=tempresult.x # Get the optimal value after updating
        finalfun=tempresult.fun
        print(f"tempresult:{tempresult}") 
        # sleep(1) 
        print(f"newep:{newep},newp:{newp}") 
        if computgap(ep, q, newep, newp) < gap:  # Training termination conditions
             break
        ep[:] = newep[:] #Update parameter values
        q[:] = newp[:]
        step = step + 1  # Perform the next iteration
    return newp,newep,finalfun # Return the final result and the final evaluation value



def GA_train(perdatalist,ep,q,epsilon_set,domain):
    """
    This training model solves the parameters in the EM model by solving 
    the numerical solution, not by analytical solution.
    """
    L=len(q) 
    # q=predealq(q)
    # ep,q=preinitial(len(ep),len(q))
    # ep=Getrandprob(len(ep))
    # q=Getrandprob(len(q))
    # ep=predealq(ep)
    bound=Bounds([0]*L, [1]*L)
    ep=list(ep)
    q=list(q) 
    gap=10**(-10) 
    maxsteps=10 
    newep=[]
    newp=[] 
    step=1 
    L2=-len(perdatalist) 
    final_resulte=0.0
    while step < maxsteps:
        print(f"step:{step},ep:{ep},q:{q}")  
        newep = updatep(perdatalist, q, ep, epsilon_set, domain) 
        tempresult=minimize(objfun,q,args=(ep,perdatalist,epsilon_set,domain,L2),method='SLSQP',constraints=cons,bounds=bound) #优化函数
        newp=tempresult.x 
        final_resulte=tempresult.fun 
        print(f"tempresult:{tempresult}") 
        # sleep(1) # 
        print(f"newep:{newep},newp:{newp}") 
        if computgap(ep, q, newep, newp) < gap:  
             break
        ep[:] = newep[:] 
        q[:] = newp[:]
        step = step + 1
    return final_resulte 





def contraints(x,n_p):
    """
    This function is mainly used to define constraints between variables.
    vars:Dictionary type contains all kinds of constraints.
    Format: {'p':[],ep:[]}
    """
    p=x[:n_p]
    ep=x[n_p:] 
    constrain1=[sum(p)-1]
    constrain2=[sum(ep)-1]
    return tuple(constrain1 + constrain2)
   


# x = np.array([0.2, 0.8, 1.2])
# bounds_min = np.zeros_like(x)
# bounds_max = np.ones_like(x)

# in_bounds = np.all(np.logical_and(x >= bounds_min, x <= bounds_max))
# print(in_bounds)

def newobjfun(x,n_p,perbdatalist,epsilon_set,domain):
    """
    The role of this function is to construct the objective function based on the perturbed data
    # var:Dictionary type for the hidden variables of the EM model {'p':[],'ep':[]}
    x: is the set of hidden variables that we booked in advance where x[:n_p] is an estimate of the true distribution of the data and x[n_p:] the distribution of users across privacy budgets
    perbdatalist: the set of data used to represent the perturbation, i.e., the set of data observed on the server side format [x_1,x_2,... ,x_N]
    epsilon_set: Used to represent the pre-agreed format of the privacy budget set between the server-side and the client-side [epsilon_1,... epsilon_t]
    domain: used to represent the value domain of the data, format [x_1,x_2,.... ,x_k]
    """
    p=x[:n_p]
    ep=x[n_p:]
    assert(len(p)==len(domain)) 
    N=len(perbdatalist) # observations number
    T=len(epsilon_set) # Number of privacy budgets

    result=0.0 #The final return on the results of the experiment.

    for i in range(N):
        for z in range(T):
            # print(f"ep[z]{ep[z]}")
            assert(ep[z]>0) # First determine if the constraints are met.
            assert(getsump(p,perbdatalist[i],epsilon_set[z],domain)>0) 
            result+=(np.log(ep[z])+np.log(getsump(p,perbdatalist[i],epsilon_set[z],domain))) 
    return -result 


"""
The purpose of this function is to add a KL scatter to the original objective function to obtain a better estimate about the p-value
"""

def KLobjfun(x,n_p,perbdatalist,epsilon_set,domain,base_ep):
    """
    base_ep: as the base distribution for KL dispersion
    """
    n=len(base_ep) 
    alpha=200000 
    KLitem=0.0
    for i in range(n):
        KLitem=KLitem+x[n_p+i]*np.log((x[n_p+i]/base_ep[i]))
    baseresult=newobjfun(x,n_p,perbdatalist,epsilon_set,domain)
    return baseresult


def KLtrain(perbdatalist,p,ep,epsilon_set,domain):
    """

    """
    gap=10**(-10) 
    maxsteps=5 # Maximum number of iterations
    newep=[] 
    newp=[] 
    # newvars=dict()
    # vars=dict()
    # vars['p']=p
    # vars['ep']=ep
    p=predealq(p)
    ep=predealq(ep)
    x_0=p+ep # Concatenate the two parameters
    print(f"x_0:{x_0}")
    finalfun=0.0
    finadistub=x_0
    Bound= bound=Bounds([1e-8]*len(x_0), [1]*len(x_0))
    step=1
    constraint = {'type': 'eq', 'fun': contraints, 'args': (len(p),)}
    while step<maxsteps:
        res=minimize(KLobjfun,x_0,args=(len(p),perbdatalist,epsilon_set,domain,ep),constraints=[constraint],bounds=Bound)
        print(f"step:{step},objfun:{res}")
        x_0[:]=res.x[:] #Update parameters, iterate over the model
        finadistub[:]=res.x[:]
        finalfun=res.fun
        step=step+1
    return finadistub,finalfun
    
    




def GAEMtrain(perbdatalist,p,ep,epsilon_set,domain):
    """
    In this method, all parameters are solved numerically, not analytically.
    perbdatalist:set of perturbed data
    epsilon_set:the set of privacy budgets for the convention
    domain:the domain of values of the data
    p:initial value of the original distribution frequency of the data
    ep:the initial value of the distribution frequency on each privacy budget
    return obj,p,ep,the final function value solution obtained by EM iteration, the estimate of the 
    distribution frequency of the original data, the estimate of the distribution frequency of the user 
    under each privacy budget
    """
    gap=10**(-10) 
    maxsteps=100 
    newep=[] 
    newp=[] 
    # newvars=dict()
    # vars=dict()
    # vars['p']=p
    # vars['ep']=ep 
    x_0=p+ep 
    print(f"x_0:{x_0}")
    Bound=Bounds([1e-8]*len(x_0), [1]*len(x_0))
    step=1
    constraint = {'type': 'eq', 'fun': contraints, 'args': (len(p),)}
    while step<maxsteps:
        res=minimize(newobjfun,x_0,args=(len(p),perbdatalist,epsilon_set,domain),constraints=[constraint],bounds=Bound)
        print(f"step:{step},objfun:{res}")
        x_0[:]=res.x[:] #Update parameters, iterate over the model
        step=step+1
    
    # return x_0


def constraint1(x):
    """
    Constraints on the variable x
    """
    return sum(x)-1


def EMtrain(perbdatalist,p,ep,epsilon_set,domain):
    """
    This function is mainly used to implement the EM algorithm for updating parameters
    perbdatalist:set of perturbed data
    p:distribution of the original data
    ep:distribution of users on each privacy budget
    epsilon_set:the given set of privacy budgets
    domain:the value domain of the data
    """
    gap=10**(-10) 
    maxsteps=5
    newep=[] 
    newp=[] 
   
    bounds=Bounds([1e-6]*len(p),[1]*len(p)) 
    step=1 
    constraint = {'type': 'eq', 'fun': constraint1} 
    while step < maxsteps: #Iterative update of parameters
        newep = updatep(perbdatalist, p, ep, epsilon_set, domain)  # Update the distribution of users across privacy budgets












# res = minimize(objective_fun, var0, args=(a, b), 
#                constraints={'type': 'ineq', 'fun': constraint_fun}, bounds=[(0, 1)]*5+ [(-1, 1)]*3)

def UPRappor(perbdata, domain, epsilon_set, ep, p):
    """
    本函数主要是用于实现在编码情况下的EM优化提升训练
    perbdata:扰动的数据经过编码之后的数据
    domain:数据的取值域
    epsilon_set:供用户选择的隐私预算的集合
    ep:初始的用户在各个隐私预算上的分布情况
    p:初始的对原始数据各个取值上的分布估计,这个先部分归一化之后的结果
    """
    K = len(domain)  # 取值域的大小
    # print(perbdata)
    # K=len(perbdata[0])
    N = len(perbdata)  # 观测数据的个数
    newp = [0] * K # 
    count=[0]*K # 用于存储使用EM算法校正之后的数量
    L=len(ep)
    for i in range(L):
        if(ep[i]<=0):
            ep[i]=0.05
    totalsum=sum(ep)
    ep=[v/totalsum for v in ep] #处理一下存在零的情况
    newepsilonset=[epsilon/2 for epsilon in epsilon_set] # 使用onehot对数据进行编码，每一位上消耗的隐私预算减半
    # randindex=rd.randint(0,K-1) #为了加快最终的求解速度，随机选择其中一个维度的数据进行EM算法的训练
    domain =[0,1] #对于编码的情况下，对数据域进行了压缩
    for index in range(K): # 对扰动数据进行按位校正
        pdata=[] # 存储需要校正的位数
        for i in range(N):
            pdata.append(perbdata[i][index])
        q=[1-p[index],p[index]] # 原始分布的初始值
        newp,_,_=newtrain(pdata,ep,q,newepsilonset,domain) #这里需要调试一下最后的步骤
        count[index]=newp[1]*N # 对第index位校正的结果
        print(count[index]) #输出一下预测的结果
    
    total=sum(count)
    return [d/total for d in count] #最终数据的分布结果

    # for i in range(N):
    #     pdata.append(perbdata[i][randindex]) #收集训练的数据
    # domain =[0,1] # 编码之后将数据的取值域映射到0，1空间内
    # q=[1-p[randindex],p[randindex]] #原始的真实的概率
    # finalldisturb,_=KLtrain(pdata,q,ep,newepsilonset,domain) #
    # return finalldisturb # 将学习到的分布返回


def UPKrr(perdata, domian, epsilon_set, ep, p):
    """
    本函数的作用是针对离散的个性化多元随机扰动的未经过编码的数据进行EM优化
    perdata:观测数据集合
    domian:数据的取值域
    epsilon_set:供用户选择的隐私预算集合
    ep:对用户在各个隐私预算上分布的初步估计
    p:对原始数据上数据分布的初始估计
    """
    print(f"domain:{domian}")
    # newq = train(perdata, ep, p, epsilon_set, domian) #这里画一下图
    newq,finalfun=newtrain(perdata,ep,p,epsilon_set,domian) #调用新的约束函数
    return newq,finalfun  # 返回最终的预测结果


def Getrandprob(k):
    """
    """
    nlist=list()
    qsum=0.0
    for i in range(k):
        temp=rd.randint(1,100)
        qsum+=temp
        nlist.append(temp)
    return [v/qsum for v in nlist]



def newUpkrr(perdata,domian,epsilon_set,epsilon_distb,Estimate_disturb):
    n1=len(domian) #数据取值域的个数
    n2=len(epsilon_set) #隐私预算的个数
    p=predealq(Estimate_disturb)
    ep=predealq(epsilon_distb)
    GAEMtrain(perdata,p,ep,epsilon_set,domian)


def KLUpKrr(perdata,domian,epsilon_set,epsilon_distb,Estimate_disturb):
    """
    添加散度之后的值目标优化函数
    """
    disturb,fianlfun=KLtrain(perdata,Estimate_disturb,epsilon_distb,epsilon_set,domian)
    return disturb,fianlfun
