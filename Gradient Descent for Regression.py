"""
COMP 551 (Applied Machine Learning) Assignment 1 Question 2
"GRADIENT DESCENT FOR REGRESSION"
Name: RASHIK HABIB
McGill University
Date: 26th January, 2018
"""

import numpy as np
import matplotlib.pyplot as plt

#############################   DATA PROCESSING  ##############################
input_train = np.genfromtxt('Dataset_2_train.csv', dtype=float, delimiter=',', usecols=0)
output_train = np.genfromtxt('Dataset_2_train.csv', dtype=float,delimiter=',', usecols=1)

input_valid = np.genfromtxt('Dataset_2_valid.csv', dtype=float, delimiter=',', usecols=0)
output_valid = np.genfromtxt('Dataset_2_valid.csv', dtype=float, delimiter=',', usecols=1)

input_test = np.genfromtxt('Dataset_2_test.csv', dtype=float, delimiter=',', usecols=0)
output_test = np.genfromtxt('Dataset_2_test.csv', dtype=float, delimiter=',', usecols=1)

#################################   VARIABLES    ##############################
print_on = 1
plot_on = 1

degree = 1
examples = 300
step_size = 0.0001
tol = 0.001
w = np.array([[5.0],[9.0]]) #initial guess
w_diff = np.array([[2.0], [2.0]]) #arbitrarily chosen for initialization
iteration = 0
Train_Errors = []
Valid_Errors = []

##################################    FUNCTIONS    ############################
#Mean Squared Error
def MSE(input_, output_, w):
    errors = np.polynomial.polynomial.polyval(input_, w)[0] - output_
    return ((errors**2).sum())/(input_.shape)

##################################   MAIN PROGRAM    ##########################
#Online Stochastic Gradient Descent
X = np.c_[np.ones(examples), input_train]
Y_train = np.c_[output_train]

while np.linalg.norm(w_diff) >= tol:
    #store old values    
    w_0 = np.copy(w[0])
    w_1 = np.copy(w[1])
    
    #one epoch
    for i in range(0, 650):
        local_error = (np.dot(X[i], w)) - Y_train[i]
        w[0] = w[0] - step_size*(local_error) 
        w[1] = w[1] - step_size*(local_error)*(X[i,1])
    
    
    w_diff = np.array([w[0] - w_0, w[1] - w_1])
    
    train_error = MSE(input_train, output_train, w)
    valid_error = MSE(input_valid, output_valid, w)
    
    
    Train_Errors.append(train_error)
    Valid_Errors.append(valid_error)

#Learning curve
plt.plot(Train_Errors, 'r-')
plt.plot(Valid_Errors, 'g-')
plt.xlabel('epoch')
plt.ylabel('error')
plt.title('Learning Curve')
plt.legend(['training error', 'validation error'], loc=4)
plt.axis([0.0, 700, 0.0, 30])
plt.show()

#Evaluate predicted y values for learnt parameters w 
x_range = np.linspace(-0.2, 1.6, 200)
Y_predicted = np.polynomial.polynomial.polyval(x_range, w)[0]
    
#################################   PRINT    ##################################
if print_on:
    print(train_error)
    print(valid_error)
    print(MSE(input_test, output_test, w))

################################   PLOTS    ###################################
if plot_on:
    #plot predicted curve along with data points - for training set
    plt.plot(input_train, output_train, 'ro')
    plt.plot(x_range, Y_predicted, 'g-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Stochastic Gradient Descent - Training set')
    plt.legend(['data points', 'predicted'], loc=2)
    plt.axis([0.0, 1.5, 0.0, 14.0])
    plt.show()
    