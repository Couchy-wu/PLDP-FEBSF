"""
The main role of this paper is to obtain the distribution of users over each 
privacy budget, given a given privacy budget, combined with a Gaussian distribution
"""
import numpy as np
import matplotlib.pyplot as plt


def getdisturb(epsilon_set):
    """
    epsilon_set:Given a collection of privacy budgets
    """
    n=len(epsilon_set)
    delta=(epsilon_set[1]-epsilon_set[0])/2
    mean=sum(epsilon_set)/n 
    # print(mean)
    std_dev=0.3 
    num_samples=3000 
    samples=np.random.normal(mean,std_dev,num_samples) 
    nums=[1]*len(samples)
    # plt.scatter(samples,nums)
    # print(samples)
    counts=[0]*n 
    for data in samples:
        for i in range(n):
            if data>=epsilon_set[i]-delta and data <=epsilon_set[i]+delta:
                counts[i]=counts[i]+1
                break
    # print(counts)
    tatalnum=sum(counts)


    return [ d / tatalnum for d in counts] # Return to distribution of end-users

# import numpy as np
# import matplotlib.pyplot as plt




def power_low_disturb(data_points):
    """Generating a power law distribution from a given set of privacy budgets"""
    # Parameters of the power law distribution
    alpha = 0.6
    # data_points = [0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,1.0]

    # Calculate the normalization constant C
    C = (alpha - 1) * (min(data_points)**(1 - alpha))

    # Calculate the probability density for each data point
    pdf = [C * x**(-alpha) for x in data_points]

    # Normalize the probabilities so they sum up to 1
    # pdf /= sum(pdf)
    pdf=[data/sum(pdf) for data in pdf]
    # print(pdf)

    # # Generate samples based on the probabilities
    # num_samples = 1000
    # samples = np.random.choice(data_points, num_samples, p=pdf)
    # print(samples)

    # # Plot the generated samples
    # plt.hist(samples, bins=len(data_points), density=True, alpha=0.7, color='b', align='left')
    # plt.xticks(data_points)
    # plt.xlabel('Sample')
    # plt.ylabel('Probability Density')
    # plt.title('Custom Power Law-like Distribution')
    # plt.show()
    return pdf




if __name__ =="__main__":
    epsilon_set= [0.1,0.3,0.5,0.7,0.9] 
    # user_disturb=getdisturb(epsilon_set)
    # print(user_disturb)
    epsilon_disturb=power_low_disturb(epsilon_set)
    print(epsilon_disturb)


