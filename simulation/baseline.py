# -------------------------------------------------------------------------
# Appendix for Holt, Kwiatkowski & Sullivan (2024)
#
# The following python code performs the Monte Carlo simulations described 
# in Holt, Kwiatkowski, & Sullivan (2024). Data from these simulations was
# used to produce Figure 1.
#
# -------------------------------------------------------------------------

# Baseline specification:
## T=3 treatment groups
## n=4 observations per treatment group

import timeit
import multiprocessing as mp
import numpy as np
from itertools import combinations, chain
from scipy.special import comb
import time

# Time the code
start_time = timeit.default_timer()

# Random number generator
rng = np.random.default_rng(4344)

# Error distributions

## f1 is Normal with mean 0 and variance of 60
def f1(size):
    return rng.normal(0,60,size)

## f2 is Uniform on [-110,110]. 
## This means it has mean 0 and variance 3333
def f2(size):
    return rng.uniform(-110,110,size)

## f3 is a 5% chance of a normal with mean 0 and variance 130,
## and a 95% chance of a normal with mean 0 and variance 45
## This distribution represents occasional outliers
## Overall, the mean is 0 and the variance is 49.25
def f3(size_):
    n = 1
    for i in size_:
        n *= i
    e1 = rng.normal(0,45,n)
    e2 = rng.normal(0,130,n)
    choice = rng.binomial(1,0.95,size=n)
    e = choice*e1 + (1-choice)*e2
    e = np.reshape(e,size_)
    return e

## f4 is a normal with mean 0 and variance 35 added to an 
## independent uniform on [-80,80]
## Overall, this is mean 0 and variance 2133
def f4(size):
    return rng.normal(0,35,size) + rng.uniform(-80,80,size)

## f5 is a logistic distribution with location 0 and scale 30
## This means the mean is 0 and the variance is 2961
def f5(size):
    return rng.logistic(0,30,size)

## f6 is a Cauchy distribution with location 0 and scale 7
## This means the mean is 0 and the variance is undefined
def f6(size):
    return 7*rng.standard_cauchy(size)

## f7 is a gumbel distribution with location -23 and scale 40
## This means it has mean 0 and variance 2632
def f7(size):
    beta = 40
    mu = -beta*0.5772
    return rng.gumbel(mu,beta,size)

## f8 is an exponential distribution with rate 0.0167 (that is, scale 60)
## Then I subtracted 60 to bring the mean down to zero
## Overall, this is mean 0 and variance 3600
def f8(size):
    return rng.exponential(60,size) - 60

## Collecting the distribution functions into a list
Fs = [f1,f2,f3,f4,f5,f6,f7,f8]

# This generator iteratively returns the next permutation of an index array
## The index array, idx, is (n x T), where 
## n is the number of data points per treatment group
## and T is the number of treatment groups
## So the index array is the size of a single sample
## Right now this only works for T = 3
def findpermutations(idx):
    n = idx.shape[0]
    T = idx.shape[1]
    f = list((idx.T).flatten())
    P = combinations(f,n)
    for i in range(int(comb(len(f),n))):
        p = next(P)
        f_ = [x for x in f if x not in p]
        Q = combinations(f_,n)
        for j in range(int(comb(len(f_),n))):
            q = next(Q)
            r = [x for x in f if (x not in p and x not in q)]
            idx_perm = np.array([p,q,r]).T
            yield idx_perm

# This function returns the Jonckheere test statistics (J) and the 
## Directional Difference test statistics (D) for an array of samples, A
## The array A is (N x m x n x T), where
## N is the number of runs in the Monte Carlo simulation
## m is the number of true treatment effects (here I use 100)
## n is the number of data points per treatment group
## T is the number of treatment groups
## So this function returns an array D and an array J, each
## of which are (N x m)
def teststats(A):
    shape = A.shape
    N = shape[0]
    m = shape[1]
    n = shape[2]
    T = shape[3]
    
    D = np.zeros((N,m))
    J = np.zeros((N,m))
    for i in range(T-1):
        for j in range(n):
            D += np.sum(A[:,:,:,i+1:]-np.expand_dims(A[:,:,j,i],axis=(2,3)),(2,3))
            J += np.sum(A[:,:,:,i+1:]>np.expand_dims(A[:,:,j,i],axis=(2,3)),(2,3))
    return D,J

# The worker function takes a permutation array, perm, and uses it to
## reindex the data, z. Then it calls teststats to get the test 
## statistics for the permuted data. All the master needs to know is 
## whether the test stats of the permuted data are more extreme than
## those of the actual data.
def worker(j, perm, answersD, answersJ):
    shape = z.shape
    N = shape[0]
    m = shape[1]
    n = shape[2]
    T = shape[3]
    
    # Need to re-index z by perm and then re-calculate D and J
    index1 = np.expand_dims(np.arange(N),axis=(1,2,3))
    index2 = np.expand_dims(np.arange(m),axis=(0,2,3))
    index3 = np.expand_dims(perm//T,axis=(0,1))
    index4 = np.expand_dims(perm%T,axis=(0,1))
    index = (index1,index2,index3,index4)
    Di,Ji = teststats(z[index])
    
    # Then return whether they are more or less extreme than the original D and J
    # Each of these should be (N x m)
    # Di and Ji are each 32 MB if N = 40000, but z[index] is 288 MB
    answersD.append((Di >= D))
    answersJ.append((Ji >= J))
    return

# The master function takes a particular error distribution (cdf)
## and creates the data. Then it iteratively creates permutation
## arrays and passes them to the workers so that the workers can
## permute the data and get the test stats of the permuted data.
## Once the master has all the permuted test stats, they can sum
## up to get a p-value for each of the m true treatment effects
## and each of the N Monte Carlo runs.
## Then the master can see in how many of the N runs there was a 
## p-value smaller than 0.05, and thus get the power of the test
## for each of the m true treatment effects.
def master(N,m,n,cdf):
    
    if __name__ == "__main__":
        
        T = 3 
        e = np.expand_dims(cdf((n,T,N)),axis=3)
        d = np.expand_dims(np.transpose(np.array(
            [[y*x for y in range(T)] for x in range(m)])),axis=(0,2))
        global z
        z = 75 + d + e
        z = np.transpose(z,axes=(2,3,0,1)) #Now z is (N x m x n x T)

        # Calculate the test statistics for the actual data
        global D,J
        D,J = teststats(z) #each should be (N x m)

        # Number of permutations
        global num_p
        num_p = int(comb(n*T,n)*comb(n*(T-1),n)) #If n = 4, T = 3, this is 34650
        
        # Permutations of index array to use for indexing
        idx = np.arange(n*T)
        idx = np.reshape(idx,(n,T))
        Permutations = findpermutations(idx)
        
        # Now pass off to workers
        ## I don't want loads of processes running at the same time on the same core
        ## I think I will join the processes in groups of 11, since 34650 is divisible by 11
        ## This makes the code a little bit hard to see through
        
        pvaluesD = np.zeros((N,m))
        pvaluesJ = np.zeros((N,m))
        for i in range(int(num_p/11)): #For each group of 11 permutations (there are 3150 such groups)
            manager = mp.Manager()
            answersD = manager.list()
            answersJ = manager.list()
            if i%10 == 0:
                print(i)
            jobs = []
            for j in range(11): #For each permutation in the group
                perm = next(Permutations)
                p = mp.Process(target=worker, args=(j, perm, answersD, answersJ))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()

            for j in range(11):
                pvaluesD += answersD[j]
                pvaluesJ += answersJ[j]
            manager.shutdown()
                
        pvaluesD = pvaluesD/num_p
        pvaluesJ = pvaluesJ/num_p
        PowerD = ((pvaluesD <= 0.05).sum(axis=0))/N
        PowerJ = ((pvaluesJ <= 0.05).sum(axis=0))/N
        
    return [PowerD,PowerJ]


    
# Graphing
import matplotlib.pyplot as plt

dist_names = ['Normal Distribution \n mean = 0, variance = 60', 
              'Uniform Distribution \n lower bound = -110, upper bound = 110',
              'Normal Distribution with Outliers \n 95% chance: Normal, mean = 0, variance = 45 \n 5% chance: Normal, mean = 0, variance = 130',
              'Sum of Normal and Uniform Shocks \n Normal, mean = 0, variance = 35 \n Uniform, lower bound = -80, upper bound = 80',
              'Logistic Distribution \n location = 0, scale = 30',
              'Cauchy Distribution \n location = 0, scale = 7',
              'Type I Extreme Value (Gumbel) Distribution \n location = -23, scale = 40',
              'Exponential Distribution \n location = -60, scale (inverse rate) = 60']

# Run the power calculation for each error distribution
BigList = []
for i in range(len(Fs)):
    print("cdf: ", i)
    A = master(40000,100,4,Fs[i]) #Should be N = 40000 for actual results
    print(A) #[D,J]
    BigList.append(A)

    title = dist_names[i]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(A[0], label='Directional Difference')
    ax.plot(A[1], label='Jonckheere')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('True Treatment Effect')
    ax.set_ylabel('Power of the Test')
    ax.set_title(title)
    ax.set_ylim([-0.1,1.1])
    #graphpath = 'Baseline/Graphs/'+str(i)+'.png'
    #plt.savefig(graphpath, bbox_inches = "tight")
    plt.show()

elapsed = timeit.default_timer() - start_time
print(elapsed)

# Saving data
# Col_list = []
# Col_list.append(pd.Series(range(100),name='d'))
# for i in range(8):
#     name_i1 = str(i)+' D'
#     name_i2 = str(i)+' J'
#     Col_list.append(pd.Series(BigList[i][0],name=name_i1))
#     Col_list.append(pd.Series(BigList[i][1],name=name_i2))

# df = pd.concat(Col_list,axis=1)
# df.to_csv('Baseline/Graphs/data.csv',index=False)