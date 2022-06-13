import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd 
def initialize(X, k):    #We assign shapes and values to the variables we set for initialization. 
    shape = X.shape      #then according to the values in our data, the number of classes and iterations, and
    n, m = shape         #we will update the values of these variables according to the operations in the functions we define these variables
    phi = np.full(shape=k, fill_value=1/k) 
    weights = np.full( shape=shape, fill_value=1/k)
    random_row = np.random.randint(low=0, high=n, size=k)
    mu = [  X[row_index,:] for row_index in random_row ]
    sigma = [ np.cov(X.T) for _ in range(k) ]
    return phi,weights,mu,sigma

def e_step(X,k,phi,weights,mu,sigma):
    # E-Step: Updates weights and phi values, keeping mu and sigma values constant
    weights = predict_proba(X,k,phi,weights,mu,sigma)
    phi = weights.mean(axis=0)
    return phi,weights,mu,sigma

def m_step(X,k,phi,weights,mu,sigma):
    # M-Step: Updates mu and sigma values, keeping weights and phi values constant
    for i in range(k):  
        weight = weights[:, [i]]
        total_weight = weight.sum()
        mu[i] = (X * weight).sum(axis=0) / total_weight
        sigma[i] = np.cov(X.T, 
            aweights=(weight/total_weight).flatten(), 
            bias=True)
    return phi,weights,mu,sigma

def fit(X, max_iter, k):
    phi,weights,mu,sigma=initialize(X,k)
    for iteration in range(max_iter): #until we reach the maximum iteration we set:
        phi,weights,mu,sigma=e_step(X,k,phi,weights,mu,sigma) #Calculating phi,weights,mu,sigma values based on e_step function
        phi,weights,mu,sigma=m_step(X,k,phi,weights,mu,sigma) #Calculating phi,weights,mu,sigma values based on m_step function
    return phi,weights,mu,sigma

def predict_proba(X,k,phi,weights,mu,sigma):
    n, m = X.shape
    likelihood = np.zeros( (n, k) )
    for i in range(k):                                #up to the number of classes with the multivariate_normal function from the scipy library
        distribution = multivariate_normal(           #first we calculate the distribution value with the help of mu and sigma values
            mean=mu[i],                               #Then we calculate the logarithmic calculation of this value with the pdf method.
            cov=sigma[i])
        likelihood[:,i] = distribution.pdf(X)

    numerator = likelihood * phi
    denominator = numerator.sum(axis=1)[:, np.newaxis]
    weights = numerator / denominator
    return weights

def predict(X,k,phi,weights,mu,sigma):
    weights = predict_proba(X,k,phi,weights,mu,sigma) #Determines the weights value based on the value found in the predict_proba function
    return np.argmax(weights, axis=1)





data=pd.read_csv("Mall_Customers.csv")
X= data.loc[:, ['Annual Income (k$)',                   #We will aggregate annual income and expenditure score data
                 'Spending Score (1-100)']].values      #We pull the values of these two columns from the dataset


#Fit a model:


k=6  #We specify the number of classes with the variable k
phi,weights,mu,sigma = fit(X,10,k)  #We determine phi,weights,mu,sigma values according to our X: data, maximum number of iterations and clustering number
prediction = predict(X, k,phi,weights,mu,sigma ) #We assign the calculations in the predict function to the prediction variable
#Plot clusters, each color indicates a class found by GMM

def plot_axis_pairs(X, prediction):
    plt.title('GMM Clusters') #Graphic title
    plt.xlabel('Annual Income (k$)') #Text on the x-axis of the graphic
    plt.ylabel('Spending Score (1-100)') #Text on the y-axis of the graphic
    plt.scatter( #Plot graphic
        X[:,0],  #Values in column 1
        X[:,1],  #Values in column 2
        c=prediction, 
        cmap=plt.cm.get_cmap('brg'), #coloring
        marker='x') #how to display datapoints
    plt.show() #show graphic
    

plot_axis_pairs(X, prediction) #call the plotting function
















