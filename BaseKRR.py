import random
import numpy as np
import soucedisturb as sdb
import dataset.datadeal as ddl
import matplotlib.pyplot as plt

epsilon=0.1

def KRR(data, epsilon, domain):
    """
    data: initialized data
    epsilon: privacy parameter
    domain: data domain
    return: the data after perturbation
    """
    n = len(data)  
    result =[data[0]]*n
    m = len(domain)  
    for i in range(n):
        p = np.exp(epsilon) / (np.exp(epsilon) + (m - 1))
        if random.random() < p:
            result[i] = data[i]
        else:
            candidate = [x for x in domain if x != data[i]]
            # print(f"data:{data[i]},perb:{result[i]}")
            result[i] = random.choice(candidate)
    return result


def perdatacount(data, domain):
    """
    data 为经过krr扰动之后的数据,data 是list类型的数据
    domain 为数据域，domain 是list 类型的数据
    return 各个取值的个数 返回值为字典类型的数据
    """
    count_dict = dict.fromkeys(domain, 0)
    for item in data:
        if item in count_dict:
            count_dict[item] += 1
        else:
            print("erro data!")
    return count_dict


def correctdatacont(dictdata, epsilon, domian):
    """
    This function mainly corrects the data after perturbation
    dictdata is a dictionary structure {value: count}.
    epsilon is the differential privacy parameter
    domian is the data domain
    """
    n = 0
    m = len(domian)
    for item in dictdata:
        n = n + dictdata[item]
    p = np.exp(epsilon) / (np.exp(epsilon) + m - 1)
    q = 1 / (np.exp(epsilon) + m - 1)
    # print(f"n:{n},p:{p},q:{q}")
    datalist = [(dictdata[item] - n * q) / (p - q) for item in dictdata]  
    return datalist


def normlizedata(datalist):
    """
    This function is mainly to normalize the data
    """
    sum = 0
    for d in datalist:
        sum += d
    normlist = [v / sum for v in datalist]
    return normlist


def BKRR(datalist, domain, epsilon):
    """
    This function serves to implement all the processes of the basic KRR algorithm
    datalist: the set of raw data
    domain:the value domain of the data.
    epsilon:privacy budget
    """
    perbdatalist = KRR(datalist, epsilon, domain)
    count_dict = perdatacount(perbdatalist, domain)
    countlist = correctdatacont(count_dict, epsilon, domain)
    disturb = normlizedata(countlist)
    return disturb # Returns the final result


def compare_experiment(k):
    """
    """
    path="./dataset/adult.data"
    # array=ddl.read_data(path)
    array=ddl.readdatafrombank("dataset/bank.csv")
    datalist,domain=ddl.gettopkvalues(array,k)
    domain=sorted(domain) 
    countlist=sdb.GetRawdatacount(datalist,domain)
    sourcedis=sdb.Getrawdisturb(countlist)
    disturb=BKRR(datalist,domain,epsilon)
    print(f"sourcedis:{sourcedis}\n")
    print(f"disturb:{disturb}\n")
    file1.write(str(sourcedis))
    file1.write("\n")
    file2.write(str(disturb))
    file2.write("\n")




if __name__ == "__main__":
    
    k=int(input("please input k:"))
    file1=open(f"./basekrrresult1/turedisturb_{k}.txt","+a")
    file2=open(f"./basekrrresult1/basekrr_{k}.txt","+a")
    for i in range(100):
        compare_experiment(k)
    file1.close()
    file2.close()


    


