"""
The role of this function module is mainly used to obtain the frequency distribution of the original data.
"""
import numpy as np
import random as rd
import scipy as sp
import sys
import os


def GetRawdatacount(sourcedata, datadomain):
    """
    The role of this function is mainly based on the given original data and data set, to obtain the number of values on each value field.
    sourcdata:raw data
    datadomain:Value range of the data
    return countlist:List of values on each value field
    """
    my_dict = dict()
    n = len(datadomain)
    for i in range(n):
        my_dict[datadomain[i]] = i  
    countlist = [0] * n  
    N = len(sourcedata) 
    for i in range(N):
        if sourcedata[i] in my_dict.keys():
            countlist[my_dict[sourcedata[i]]] += 1  # count plus one
    return countlist  # Return the final result



def Getrawdisturb(countlist):
    """
   This function is mainly used to realize the frequency distribution of obtaining the original data.
    return: rawdisturb
    """
    sum = 0
    for num in countlist:
        sum += num

    rawdisturb = [v / sum for v in countlist]
    return rawdisturb  # Returns the frequency distribution of the original data
