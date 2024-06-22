"""
This function is mainly used to implement user-level personalized multivariate stochastic perturbation mechanism
"""
import numpy as np
import pandas as pd
import random
import sys
import BaseKRR as bk
import modal
import dataset.datadeal as dl
import soucedisturb as sdb
from time import sleep
import EMalgorithm as EM
import sample as sp


epsilon_set = [0.1,0.2] # Set of privacy budgets agreed upon in advance
epsilon_p = 0.2 # Implementation of the privacy budget agreed upon to protect the user's privacy budget
disturb_user = [0.8,0.2] # Frequency distribution of users on the following privacy budgets



def baseperbdata(data, epsilon, domain):
    """
    This function mainly accomplishes the perturbation of a single piece of data
    data:a single piece of data
    epsilon:the given privacy budget
    domian:the given data domain
    """
    m = len(domain)
    p = np.exp(epsilon) / (np.exp(epsilon) + m - 1)
    if random.random() <= p:
        result = data
    else:
        candidate = [x for x in domain if x != data]
        result = random.choice(candidate)
    return result


def PKRRdata(data, epsilon_set, domian, epsilon_p, disturb_user):
    """
    This function is mainly a multivariate stochastic perturbation mechanism to personalize the user data
    Firstly, the user samples the collection of privacy budgets from the set according to the user frequency defined by the realization, and after the completion of sampling
    The user's data is first perturbed using the selected privacy budget.
    Then the selected privacy budget is perturbed
    After the above perturbation, send the perturbed privacy budget and data to the server.
    data: is the list of user's raw data
    epsilon_set: the set of reserved privacy budgets.
    domain: the user's data domain.
    disturb_user: the distribution of the user's selected privacy budgets.
    """
    n = len(data)
    sample = modal.simple_epsilon(epsilon_set, disturb_user, n)  # Completion of user sampling of privacy budgets
    # print(sample)
    newdatalist = []
    newepsilonlist = []
    if len(sample) != n:
        print("erro !!!")
    for i in range(n):
        d = baseperbdata(data[i], sample[i], domian)
        e = baseperbdata(sample[i], epsilon_p, epsilon_set)
        newdatalist.append(d)
        newepsilonlist.append(e)

    return newdatalist, newepsilonlist  # Return user data after perturbation and privacy budget after perturbation


def aggregat_data(datalist, domian):
    """
    This function is mainly for the list of elements in accordance with the same value of the statistics
    datalist:data set
    domian:data field
    return :dictionary collection dict data personality {data:count}
    """
    counter = {}
    for item in domian:
        counter[item] = 0
    for d in datalist:
        if d in counter:
            counter[d] += 1
        else:
            print("data errorr!!!")
    return counter


def u_aggeraget_epsilon(epsilon_dict, epsilonp, epsilon_set):
    """
    This method is mainly used to estimate the frequency distribution of users on each privacy budget
    epsilon_dict:number of users on each privacy budget {epsilon:count}
    epsilonp:privacy budget used to protect user privacy
    epsilon_set:set of privacy budgets that implement the reservation
    """
    newlist = bk.correctdatacont(
        epsilon_dict, epsilonp, epsilon_set
    )  
    return nomalization_data(newlist)  # Return the frequency distribution of users across privacy budgets after normalization


def u_aggregatdata(epsilon_set, epsilon_disturb, data_dict, domian):
    """
    This function is mainly used to correct the data after perturbation using different privacy budget levels
    epsilon_set:pre-set privacy budget set [epsilon_1,.... ,epsilon_m]
    epsilon_disturb:distribution of users across privacy budgets -> [0.1,0.2.... ,0.1]
    data_dict:number of individuals corresponding to each data data value data format: {key:value}
    domiam:data domain {x_1,... ,x_k}
    """
    m = len(domian)
    if len(epsilon_set) != len(epsilon_disturb):
        print("error!!!!")  
    p = 0.0
    q = 0.0
    n = len(epsilon_set)
    # sleep(1)
    print(epsilon_disturb)
    print(epsilon_set)
    for i in range(n):
        p = p + (epsilon_disturb[i] * modal.getprobility_1(epsilon_set[i], m))
        q = q + (epsilon_disturb[i] * modal.getprobility_2(epsilon_set[i], m))
    #     print(f"p:{p},q:{q}")
    print(f"p:{p},q:{q}m:{n*q}")
    N = 0  #
    for item in data_dict:
        N += data_dict[item]
        # print(f"key;{item},count:{data_dict[item]}")

    datacount = [(data_dict[item] - N * q) / (p - q) for item in data_dict]  
    # print(datalist)
    return datacount  # Return the final corrected count


def nomalization_data(datalist):
    """
    This function is mainly to normalize the original data
    """
    sum = np.sum(datalist)
    newdatalist = [d / sum for d in datalist] 
    return newdatalist


def UPKR(datalist, domain, epsilon_set, epsilonp, disturb):
    """
    This function is mainly used to implement the personalized multivariate randomized perturbation algorithm (UPKRR)
    datalist:the original data without perturbation
    domain: the value domain of the data.
    epsilon_set: the set of privacy budgets for the user to choose from.
    disturb:the distribution of users across privacy budgets
    epsilonp:privacy budget used to protect the user's privacy budgets
    """
    newdatalist, newepsilonlist = PKRRdata(
        datalist, epsilon_set, domain, epsilonp, disturb
    )
    perdatadict = aggregat_data(newdatalist, domain)
    perepsilondict = aggregat_data(newepsilonlist, epsilon_set)
    epsilondisturb = u_aggeraget_epsilon(perepsilondict, epsilonp, epsilon_set)
    cordatacountlist = u_aggregatdata(epsilon_set, epsilondisturb, perdatadict, domain)
    return nomalization_data(cordatacountlist),epsilondisturb,newdatalist # Unbiased estimation of data and unbiased estimation of privacy budget both return


"""The main role of this function is to realize the overall flow of the experiment, repeat the experiment 100 times and store the results of the experiment in the file basekrr.txt, EMoptimal.txt, the file"""

def Comparison_experiments(path,k,epsilon_set,epsilon_p,disturb_user):
    """
    The main purpose of this function is to realize a comparison experiment between the basic module and the added EM module
    path:path of the dataset
    k:dimension of the original data
    epsilon_set:set of privacy budgets
    epsilon_p:privacy budget to protect the user's privacy budget
    disturb_user:distribution of users across privacy budgets
    """
    # array = dl.read_data(path)
    # array=dl.readdatafrombank("dataset/bank.csv")
    array=dl.readdatafromcensus("dataset/acs2017_census_tract_data.csv")
    datalist, domain = dl.gettopkvalues(array, k)
    domain = sorted(domain)
    # print(domain)
    countelist = sdb.GetRawdatacount(datalist, domain)
    sourcedisturb = sdb.Getrawdisturb(countelist)
    print("---Original true distribution results---")
    print(sourcedisturb)
    file1.write(str(sourcedisturb)) 
    file1.write("\n")
    print("---Results of the predictive distribution---")
    Estimate_disturb,epsilon_distb,perbdatalist = UPKR(datalist, domain, epsilon_set, epsilon_p, disturb_user)
    print(Estimate_disturb) # Personalized multivariate stochastic response model predictions
    file2.write(str(Estimate_disturb)) # Write the estimation results of the basic model
    file2.write("\n")
    print(epsilon_distb)
    # EMdisturb,EMfun=EM.UPKrr(perbdatalist,domain,epsilon_set,epsilon_distb,Estimate_disturb) 
    # file3.write(str(EMdisturb)+":"+str(EMfun)+"\n") 
    # print(EMdisturb) 
    # print("newtrain的情况")
    p,ep,finafun= EM.newtrain(perbdatalist,epsilon_distb,Estimate_disturb,epsilon_set,domain)
    print(f"p:{p}")
    print(f"ep:{ep}")
    file3.write(str(tuple(p))) # Write the final estimate of the EM algorithm to a file
    file3.write("\n") 
    print(f"finafun:{finafun}")
    #===================================== 
    print("EM algorithm after adding the scatter value")
    # distub,finalfun=EM.KLUpKrr(perbdatalist,domain,epsilon_set,epsilon_distb,Estimate_disturb) #修改一下这个实验结果
    # Recalibrate the final result
    index=len(Estimate_disturb)
    # EMp=distub[:index]
    EMep=[1/len(epsilon_set)]*len(epsilon_set) #=distub[index:] 
    # Verify that there's no problem with the data points
    # print(f"EMP:{EMp}")
    print(f"EMep:{EMep}")
    assert(len(EMep)==len(epsilon_set))
    datadict=aggregat_data(perbdatalist,domain)
    redirctcount=u_aggregatdata(epsilon_set,EMep,datadict,domain) 
    finaldisturb=nomalization_data(redirctcount) # return the final distribution result
    print(f"finaldisturb:{finaldisturb}") # Output final distribution values
    file4.write(str(finaldisturb)+"\n") # Write the result in the file after adding the KL dispersion




"""The following is mainly for model parameterization."""
# epsilon_set_list=[[0.1,0.2],[0.1,0.2,0.3],[0.1,0.2,0.3,0.4]]
# disturb_user_list=[[0.5,0.5],[0.4,0.3,0.3],]
# epsilon_p_list=[0.5,0.5,0.5,0.5]




def all_comparison_experiments():
    """
    First use the command line for parameter input
    """
    path ='dataset/adult.data' 
    args = sys.argv
    k=int(args[1])
    file1=open(f"./KRRCENSUS1/{k}_true_disturb.txt",'+a') #实验数据集合的真实分布
    file2=open(f"./KRRCENSUS1/{k}_basemodekrr.txt",'+a')
    file3=open(f"./KRRCENSUS1/{k}_EMoptimal.txt",'+a') #EM模型的优化结果
    file4=open(f"./KRRCENSUS1/{k}_KLEMoptimal.txt",'+a')
    steps=0
    while steps<100:
        Comparison_experiments(path,k,epsilon_set,epsilon_p,disturb_user)
        steps=steps+1
    file1.close()
    file2.close()
    file3.close()
    file4.close()



if __name__ == "__main__":
    """
    Test the implementation of the basic UPKRR algorithm
    """
    path ='dataset/adult.data' 
    # path = (
    #     "C:\\Users\\Sky\\Desktop\\personal_privacy_code\\dataset\\adult.data"  
    # )
    k=int(input()) 
    file1=open(f"./KRRCENSUS2/{k}_true_disturb.txt",'+a') 
    file2=open(f"./KRRCENSUS2/{k}_basemodekrr.txt",'+a') 
    file3=open(f"./KRRCENSUS2/{k}_EMoptimal.txt",'+a') 
    file4=open(f"./KRRCENSUS2/{k}_KLEMoptimal.txt",'+a') 
    steps=0
    # disturb_user=sp.getdisturb(epsilon_set)
    # disturb_user=sp.power_low_disturb(epsilon_set)
    print(disturb_user)
    while steps<100:
        Comparison_experiments(path,k,epsilon_set,epsilon_p,disturb_user)
        steps=steps+1
    file1.close()
    file2.close()
    file3.close()
    file4.close()



