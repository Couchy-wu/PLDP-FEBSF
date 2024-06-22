"""
This document is mainly used to realize all the public functions that need to be used in the experiment, such as data encoding, data reading, data perturbation, data sampling and other functions.
"""
import numpy as np
import pandas as pd
import random as rd
import scipy as sp
import modal as md
import sys
import os


def classify_data(datalist, epsilonlist, epsilon_set):
    """
    This function is mainly for all users to send data, according to the different levels of privacy budget for classification processing
    datalist: the set of perturbed data.
    epsilonlist: each data set and its corresponding privacy budget level.
    epsilon_set: the given set of privacy budget levels.
    """
    assert len(datalist) == len(epsilonlist)  # First, determine if the list length is consistent
    n = len(datalist)  # Length of data
    m = len(epsilon_set)  # Length of privacy budget levels
    groupdatalist = [[]] * m
    my_dict = {}
    for i in range(m):
        my_dict[epsilon_set[i]] = i
    for i in range(n):
        index = my_dict[epsilonlist[i]]
        groupdatalist[index].append(datalist[i])

    return groupdatalist  # Returns the grouped data


def getdisturb(epsilon_set, disturb, size):
    """
    The main purpose of this function is to generate data according to the specified distribution
    epsilon_set: the data set to be sampled.
    disturb:distribution of samples
    size:size of the sample
    """
    epsilonlist = md.simple_epsilon(epsilon_set, disturb, size)
    return epsilonlist


def encodedata(data, domian):
    """
    Given a data field, encode the data.
    and return the encoded data
    """
    domian=sorted(domian)
    n = len(domian)
    encode = [0] * n
    for i in range(n):
        if data == domian[i]:
            encode[i] = 1
    return encode


