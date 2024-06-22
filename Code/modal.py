"""This document is mainly used to implement the base model, which is used to construct dynamic matrices."""
import numpy as np
import pandas as pd
import matplotlib as mp
import sys
import random

a = [0.1, 0.2, 0.3] # set of privacy budgets for users
epsilonp = 2 # Privacy budget used to protect the user's privacy budget


def random_respond(f, x, data, p):
    """this function used to implement the base random_respond algrithm"""
    f = random(0, 1)
    if f <= p:
        x = x
    else:
        newlist = list(filter(lambda y: y != x, data))  # Remove element x from the original array
        x = newlist[random.randint(0, newlist.size())]  # Randomly select an element from the remaining elements for replacement


def simple_epsilon(values, probility, size):
    """In this function we specify the probability of the distribution of the user's 
    privacy budget level and sample privacy budgets from the specified set of privacy budgets according to the modified probability"""
    # values = [1, 2, 3, 4, 5]
    # probabilities = [0.1, 0.2, 0.3, 0.2, 0.2]
    # probility=probility*len(values)
    samples = random.choices(values, probility, k=size)
    return samples


def getprobility_1(epsilon, k):
    """Calculate the probability of maintaining the original value given a privacy budget"""
    p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    return p


def getprobility_2(epsilon, k):
    """Given the privacy budget, calculate the probability of perturbation to other values"""
    p = 1 / (np.exp(epsilon) + k - 1)
    return p


def encodedata(domain, response):
    """Onehot coding of multivariate discrete data, specifying element and domain spaces"""
    # domain=sorted(domain)
    encod = [1 if d == response else 0 for d in domain]
    return encod


def getobersvercount(encodelist):
    """This function is mainly for the encoding after the perturbation of the binary vector by bit counting, encodelist is a two-dimensional vector group"""
    n = len(encodelist[0])
    count = [0] * n  # Initialize a zero vector of length n for counting
    for x in encodelist:
        for i, bit in enumerate(x):
            if bit == 1:
                count[i] = count[i] + 1

    return count  


def perturb(encoded_response, p, q):
    """Scrambling of encoded data by bits"""
    return [perturb_bit(b, p, q) for b in encoded_response]


def perturb_bit(bit, p, q):
    """For each of the data"""
    sample = np.random.random()
    if bit == 1:
        if sample <= p:
            return 1
        else:
            return 0
    elif bit == 0:
        if sample <= q:
            return 1
        else:
            return 0


"""On the calibration side, it may be necessary to modify the form of the calibration."""


def aggregate(responses, p, q, n):
    """
    The role of this function is to correct the observed data for
    p:probability of retaining to the original value
    q:probability of perturbation to other values
    n:total user data
    """
    print(responses)
    print(f"n:{n},p:{p},q:{q}")
    return [(v - n * q) / (p - q) for v in responses]
