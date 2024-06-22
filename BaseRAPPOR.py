"""
This document is mainly used to implement the basic RAPPOR algorithm (in this paper, the optimal encoding method is used instead of the traditional hash encoding method)
This module provides the following functions:
Encoding of data sets
Perturbation of the encoding
Aggregation of data
Estimation of final frequencies
"""
import numpy as np
import pandas as pd
import matplotlib as plt
import modal as md
import utility as utl
import dataset.datadeal as ddl 
import soucedisturb as sdb
import random
import os


epsilon=0.2 #Setting the size of the privacy budget

def encodedata(datalist, domain):
    """
    This function is mainly onehot coding the data list
    datalist:the data to be encoded
    domian: the value range of the data.
    """
    domain=sorted(domain)
    n = len(datalist)
    encoddatalist = []
    for i in range(n):
        encoddatalist.append(md.encodedata(domain, datalist[i]))
    return encoddatalist  # Returns encoded data


def perbdata(encoddatalist, p, q):
    """
    This function mainly implements the scrambling of all encoded data
    encoddatalist:set of encoded data
    p:probability of retaining as original
    q:probability of perturbation to other values
    """
    n = len(encoddatalist)
    perbdatalist = []
    for i in range(n):
        perbdatalist.append(md.perturb(encoddatalist[i], p, q))
    return perbdatalist  # Return a collection of perturbation data


def obsercount(perbdatalist):
    """
    This function is mainly used to count the scrambled coded data
    perbdatalist:coded data after perturbation
    """
    return md.getobersvercount(perbdatalist) 


def correctdatacount(counterlist, p, q, n):
    """
    Bitwise correction of the count of the observation data.
    counter:list of bitwise counts
    p:probability of retaining the original value
    q:probability of perturbation to other values
    """
    return md.aggregate(counterlist, p, q, n)


def getdisturb(corcounterlist):
    """
    Normalize the corrected array.
    corcounterlist:Corrected counterlist
    """
    # print(f"corecounterlist1:{corcounterlist}")
    n =len(corcounterlist)
    for i in range(n):
        if corcounterlist[i]<0:
            corcounterlist[i]=0     
    # print(f"corecounterlist:{corcounterlist}")
    n = np.sum(corcounterlist)
    return [v / n for v in corcounterlist]


def BRappor(datalist, domian, epslion):
    """
    This function mainly implements the basic RAPPOR algorithm based on the given data set and the privacy budget
    """
    encodedatalist = encodedata(datalist, domian)
    # k = len(domian)
    p = md.getprobility_1(epsilon=epslion / 2, k=2)
    q = md.getprobility_2(epsilon=epslion / 2, k=2)
    perbdatalist = perbdata(encodedatalist, p, q)
    n = len(datalist)  
    conterlist = obsercount(perbdatalist)
    corcounterlist = correctdatacount(conterlist, p, q, n)
    return getdisturb(corcounterlist=corcounterlist)  # Return the final distribution

def compare_experiment(k):
    """
    """
    path="./dataset/adult.data" # Set the path of the input data to correspond to the paths of the three data files in the data file
    # array=ddl.read_data(path)
    array=ddl.readdatafrombank("dataset/bank.csv")
    datalist,domain=ddl.gettopkvalues(array,k)
    domain=sorted(domain)
    sourcecount=sdb.GetRawdatacount(datalist,domain)
    sourcedisturb=sdb.Getrawdisturb(sourcecount) # Real distribution
    perbdisturb=BRappor(datalist,domain,epsilon)
    # print(sourcedisturb)
    # print(perbdisturb)
    file1.write(str(sourcedisturb))
    file1.write("\n")
    file2.write(str(perbdisturb))
    file2.write("\n")




if __name__ == "__main__":
    """The main focus here is to implement a test of the baseRAPPOR algorithm"""
    k=int(input("please input dim k:"))
    file1=open(f"./baserapporresult1/truedisturb1_{k}.txt","+a")  #Set the output path for the true distribution of data
    file2=open(f"./baserapporresult1/baserappor1_{k}.txt","+a") # Set the output path of the corrected distribution
    for i in range(100):
        compare_experiment(k)
    file1.close()
    file2.close()






