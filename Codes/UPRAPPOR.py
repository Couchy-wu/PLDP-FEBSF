import numpy as np
import pandas as pd
import random
import sys
import dataset.datadeal
import modal
import BaseRAPPOR as bp
import dataset.datadeal as dl
import soucedisturb as sdb
import EMalgorithm as EM
from time import sleep
import sample as sp

"""
This document is mainly used to implement the RAPPOR algorithm for user-level personalized differential privacy
In this algorithm, all our data is encoded using onehot coding and the data is perturbed bit-wise
The data to be encoded includes the user's chosen privacy budget and the user's real data.
For the perturbation of the data the basic rappor method is used uniformly for perturbation.
"""

# epsilon_set = [] 
# epsilonp = 1  
# epsilon_disturb = [] 
# epsilonp = 1  

epsilon_set=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] # given a collection of privacy budgets 
disturb=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] # The set of distributions of a given user across privacy budgets
# disturb=sp.power_low_disturb(epsilon_set) 
epsilonp=0.7 # Privacy budget given the budget for protecting user privacy


def sample_epsilon(epsilon_set, n, epsilon_disturb):
    """
    This method focuses on collecting n data from the specified distribution
    epsilon_set: the set to be sampled
    n: the number of data to be sampled
    epsilon_disturb: the distribution to be sampled.
    return: return the list of samples.
    """
    epsilonlist = modal.simple_epsilon(epsilon_set, epsilon_disturb, n)
    return epsilonlist


def perb_encodedata(data, p, q):
    """
    The function is mainly used to complete the encoded data in accordance with the specified privacy budget for perturbation
    data: after encoding the data [0,0,1,0,0].
    p:probability of data[i]=1 after perturbation when data[i]=1
    q: when data[i]=0, after the perturbation of data[i]=1 probability
    """
    # print(data)
    assert(p+q==1) 
    for i in range(len(data)):
        if data[i] == 1:
            if random.random() <= p:
                data[i] = 1
            else:
                data[i] = 0
        elif data[i] == 0:
            if random.random() <= q:
                data[i] = 1
            else:
                data[i] = 0
    # sleep(1)
    # print(data)
    return data


def encodedata(data, domian):
    """
    Given a data field, encode the data.
    and return the encoded data.
    """
    # domain=sorted(domain)
    n = len(domian)
    encode = [0] * n
    for i in range(n):
        if data == domian[i]:
            encode[i] = 1
    # sleep(1)
    # print(encode) 
    return encode


def encodealldata(datalist, domian):
    """
    This function mainly encodes the specified data set onehot and returns the encoded set
    datalist: the specified set of uncoded data, format -> [x1,x2,.... ,xn]
    domian:the value range of the specified data, format->[x1,x2,.... ,xk]
    return: return the set of data after onehot coding, format->[[0,1,0,.... ,0],... ,[1,0,... ,0]]
    """
    # domain=sorted(domain)
    encodedatalist = []
    for d in datalist:
        encode = encodedata(d, domian)
        encodedatalist.append(encode)
    return encodedatalist


def personalperbdata(datalist, epsilonlist, domian):
    """
    This function is mainly to complete the personalized perturbation of the data, that is, each piece of data in accordance with the given privacy budget for perturbation
    datalist: list of raw data (uncoded set)
    domian:data domain
    epsilonlist:set of privacy budgets specified for each piece of data (uncoded set)
    return :perdatalist(set after perturbation of coding)
    """
    # domain=sorted(domain)
    encodedatalist = encodealldata(datalist, domian)
    assert len(encodedatalist) == len(epsilonlist)  
    perbadatalist = []
    for i in range(len(epsilonlist)):
        p = modal.getprobility_1(epsilonlist[i] / 2, 2)  
        q = modal.getprobility_2(epsilonlist[i] / 2, 2) 
        perbadatalist.append(perb_encodedata(encodedatalist[i], p, q))
    return perbadatalist


def perbepsilonlist(epsilon_list, epsilon_set, epsilon_p):
    """
    This function protects the user's privacy budget.
    epsilon_list:user's privacy budget list
    epsilon_set:privacy budget set
    epsilon_p:privacy budget used to protect the user's privacy budget
    """
    encode_epsilon_list = encodealldata(epsilon_list, epsilon_set)  # Start by coding the privacy budget
    n = len(encode_epsilon_list)
    per_epsilon_list = []
    for i in range(n):
        p = modal.getprobility_1(epsilon_p / 2, 2)
        q = modal.getprobility_2(epsilon_p / 2, 2)
        per_epsilon_list.append(perb_encodedata(encode_epsilon_list[i], p, q))
    return per_epsilon_list


def get_bit_count(perlistdata):
    """
    Count the scrambled coded data by bits
    perlistdata: set of 0,1 data after perturbation, data format is -> [[0,1,1,... ,0],... ,[0,0,1,... ,1]]
    return:list of counts at the corresponding position, data format -> [6,7,9,... ,10]
    """
    counterlist = modal.getobersvercount(perlistdata)
    return counterlist


def getprobility(epsilon_set, epsilon_disturbe):
    """
    This function is mainly used to calculate the probability of personalized perturbation in the method, that is, the existence of multiple privacy budget level, the probability of perturbation calculation
    epsilon_set: the set of privacy budgets for the user to choose.
    epsilon_disturbe: denotes the frequency of the user's distribution on each privacy budget.
    return : p,q p:probability of retaining as original value q:probability of other values perturbing to this value
    """
    p = 0.0
    q = 0.0
    assert len(epsilon_set) == len(epsilon_disturbe)  # Prevents inconsistencies in length and facilitates inspections
    n = len(epsilon_set)
    for i in range(n):
        p = (
            p + modal.getprobility_1(epsilon_set[i] / 2, 2) * epsilon_disturbe[i]
        ) # Perform weighted sums
        q = q + modal.getprobility_2(epsilon_set[i] / 2, 2) * epsilon_disturbe[i]
   # assert p + q == 1  
    return p, q


def correctcount(datalist, epsilon_set, epsilon_disturbe, num_users):
    """
    This function is to correct the count list by position.
    datalist:count data for each position
    epsilon_set: the set of privacy budgets for the user to choose from.
    epsilon_disturbe: the frequency of the user's distribution on each privacy budget.
    p:indicates the probability of retaining as original value
    q:probability that other values are perturbed to the original value
    num_users: the total number of users, to facilitate later correction
    return : return the corrected count data
    """
    p, q = getprobility(epsilon_set, epsilon_disturbe)  # First get two probabilities
    newdatalist = [(v - num_users * q) / (p - q) for v in datalist]  # Distribution of data after correction
    return newdatalist


def getdisturb(pertub_epsilon_datas, epsilonp):
    """
    The main purpose of this function is to obtain the distribution of the user's privacy budgets from the perturbed privacy budget data in order to correct the data at a later stage.
    epsilon_set: the set of privacy budgets for users to choose.
    pertub_epsilon_datas: the set of perturbed privacy budgets encoded.
    epsilonp: the privacy budget used to protect the user's privacy budget. Encoded privacy budget
    """
    n = len(pertub_epsilon_datas)  
    percountlist = modal.getobersvercount(pertub_epsilon_datas)
    p = modal.getprobility_1(epsilonp / 2, 2)
    q = modal.getprobility_2(epsilonp / 2, 2) 
    corcountlist = bp.correctdatacount(percountlist, p, q, n)
    return bp.getdisturb(corcountlist) 


def personaggregatdata(correctcountlist):
    """
    This function is mainly to achieve the perturbation of the data for the correction, normalization processing
    correctcountlist:correctcountlist:correctedcountlist
    """
    sum = np.sum(correctcountlist)  
    return [v / sum for v in correctcountlist]  


def uprap(datalist, epsilon_set, disturb, epsilonp, domian):
    """
    This function is mainly used to implement the basic personalized local differential privacy RAPPOR algorithm of the overall process (does not include the optimization process)
    datalist:set of raw uncoded data
    epsilon_set: the set of privacy levels provided to the user to choose from (incremental)
    disturb:the true distribution of users across privacy budgets
    epsilonp:the privacy budget used to protect the privacy budget of the user
    domian: the domain of values of the data
    return: Returns an unbiased estimate of the frequency distribution of the original data.
    """
    n = len(datalist)  # Number of users
    epsilonlist = sample_epsilon(epsilon_set, n, disturb)
    encodedatalist = personalperbdata(datalist, epsilonlist, domian)
    perbeplist = perbepsilonlist(
        epsilon_list=epsilonlist, epsilon_set=epsilon_set, epsilon_p=epsilonp
    )   # Scrambling and coding of users' real privacy budgets
    epdisturb = getdisturb(perbeplist, epsilonp)  # Obtained unbiased estimates of users on individual privacy budgets
    obcountlist = get_bit_count(encodedatalist) # Bit-by-bit counting of data after user perturbation
    corcountlist = correctcount(obcountlist, epsilon_set, epdisturb, n)
    return personaggregatdata(corcountlist),epdisturb,encodedatalist # Return an unbiased estimate of the distribution of the original data, return the perturbed data




def compareUPrappor(k):
    """
    The main purpose of this function is to correct the coded data and to perform experimental comparisons
    k:dimension of the data
    """
    # path=". /dataset/adult.data" # file path of the perturbed data set
    # array=dl.read_data(path) # Corresponding files need to be matched
    # array=dl.readdatafrombank("dataset/bank.csv")
    array=dl.readdatafromcensus("dataset/acs2017_census_tract_data.csv")
    datalist,domain=dl.gettopkvalues(array,k)
    domain=sorted(domain)
    countelist=sdb.GetRawdatacount(datalist,domain)
    sourcedisturb=sdb.Getrawdisturb(countelist)
    file1.write(str(sourcedisturb))
    file1.write("\n") #store the true distribution into the file
    datadisturb,epsilondist,perbdatalist=uprap(datalist,epsilon_set,disturb,epsilonp,domain) #After correcting the disturbed data using the dynamic correction algorithm
    file2.write(str(datadisturb))
    file2.write("\n") # write the result of the dynamic correction to the file
    finaldisturb= EM.UPRappor(perbdatalist,domain,epsilon_set,epsilondist,datadisturb) # go through EM algorithm to learn 
    print(f"EMoptimal:{finaldisturb}") 
    file4.write(str(finaldisturb))
    file4.write("\n")
    # =========================================== this part of the content as a comparative analysis of the content directly to the real user distribution of data to correct, see the final results
    finalepsilondist=disturb # finaldisturb[2:] # get the results of the distribution of learning temporarily directly with the real comparison it look at the actual results
    obcountlist = get_bit_count(perbdatalist) # count the data by bit after user perturbation
    n=len(obcountlist)
    corcountlist = correctcount(obcountlist, epsilon_set, finalepsilondist, n)
    finaldatadisturb=personaggregatdata(corcountlist)
    file3.write(str(finaldatadisturb)) #
    file3.write("\n") # debug the final process
    file1.flush()
    file2.flush()
    file3.flush()
    file4.flush()



"""Check and analyze all procedures"""
if __name__ == "__main__":
    """ Realize all processes of the algorithm."""
    for k in [2,3,4,5,6,7,8,9]:
    # k=int(input("please input the dim of data:")) # set the number of data collection attributes
        file1=open(f". /PDCAE5/truerappordistub{k}.txt", "+a") #save the original distribution of the data
        file2=open(f". /PDCAE5/basemoderappordistub{k}.txt", "+a") #Save the base model calibration results
        file3=open(f". /UPRAPPOR14/KLoptimalrappor{k}.txt", "+a") #Save the results of using EM to predict the distribution of users after correction
        file4=open(f". /UPRAPPOR14/EMrappor{k}.txt", "+a") #Save the results after optimization using the EM algorithm
        for i in range(100): 
            compareUPrappor(k)
        file1.close()
        file2.close()
    file3.close()
    file4.close()
   



