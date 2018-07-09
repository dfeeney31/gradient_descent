import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(5280)
Xs = np.random.rand(100) * 100
Ys = Xs + (np.random.rand(100) * 100)

plt.scatter(Xs,Ys)
plt.show()

def computeCost(Xs,Ys,m,theta):
    #requires the correct dimensions of Xs, which is 2  x # observations
    #Check if correct dimensions, if not, add a vector of ones
    #assert (Xs.shape == (len(Xs),2)),"Incorrect size, append zeros"
    if (Xs.shape != (len(Xs),2)):
        Xs = np.vstack((np.ones(len(Xs)),Xs)).T
        print("Adding column of ones to make correct size")
    #Calculate cost
    predicted_value = np.matmul(Xs,theta)
    sqrErrors = np.power((predicted_value - Ys),2)
    J = (1/2)*(1/m) * np.sum(sqrErrors)
    return J

#preallocate a vector of ones
one_array = np.ones(len(Xs))
#stack and transpose so Xs and combined with the ones

# true_xs = np.vstack((one_array,Xs)).T
# true_xs.shape == (len(true_xs),2)
#
# computeCost(true_xs,Ys,len(Xs),np.array([1,1]))
computeCost(Xs,Ys,len(Xs),np.array([1,1]))

#computeCost(Xs,Ys,)
theta = np.array([1,1])
#testing_t[0] + testing_t[1] * x
alpha = 0.0001
m = len(Ys)
num_iters = 100
#Xs = np.vstack((np.ones(len(Xs)),Xs)).T

def gradientDescent(Xs,Ys,theta,alpha,num_iters):
    global theta_one
    global theta_two
    theta_one = []
    theta_two = []
    m = len(Ys) #m will be used in the cost function
    global J_hist
    J_hist = [] #preallocate a vector for the cost function
    for i in range(num_iters):
        J_hist.append(computeCost(Xs,Ys,m,theta))
        x = Xs[:,1]
        predicted_value = theta[0] + (theta[1] * x)
        thetaOne = theta[0] - alpha * (1/m) * np.sum(predicted_value - Ys)
        thetaTwo = theta[1] - alpha * (1/m) * np.sum((predicted_value - Ys) * x)
        # print("theta one is", thetaOne)
        # print("Theta two is", thetaTwo)
        theta_one.append(thetaOne)
        theta_two.append(thetaTwo)
        theta = np.array([thetaOne,thetaOne])
    return [J_hist]
    plt.scatter(J_hist)
    plt.show()

# p_val = theta[0] + (theta[1] * x)
# theta[1] - alpha * (1/m) * np.sum(p_val - Ys) * x

gradientDescent(Xs,Ys,theta,alpha,1000)
plt.scatter(np.arange(0,len(J_hist)),J_hist)
plt.show()
