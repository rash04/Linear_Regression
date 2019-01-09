"""
COMP 551 (Applied Machine Learning) Assignment 1 Question 1
"MODEL SELECTION"
Name: RASHIK HABIB
McGill University
Date: 26th January, 2018
"""

import numpy as np
import matplotlib.pyplot as plt

#############################   DATA PROCESSING   #############################
input_train = np.genfromtxt('Dataset_1_train.csv', dtype=float, delimiter=',', usecols=0)
output_train = np.genfromtxt('Dataset_1_train.csv', dtype=float,delimiter=',', usecols=1)

input_valid = np.genfromtxt('Dataset_1_valid.csv', dtype=float, delimiter=',', usecols=0)
output_valid = np.genfromtxt('Dataset_1_valid.csv', dtype=float, delimiter=',', usecols=1)

input_test = np.genfromtxt('Dataset_1_test.csv', dtype=float, delimiter=',', usecols=0)
output_test = np.genfromtxt('Dataset_1_test.csv', dtype=float, delimiter=',', usecols=1)


##############################   VARIABLES    #################################
plot_on = 1
print_on = 1
regularization_on = 1 #L2 Regularization

degree = 20
examples = 50
if regularization_on:
    lamda = 0.017
else:
    lamda = 0.0

############################    FUNCTIONS    ##################################
#Mean Squared Error
def MSE(input_, output_, w):
    errors = np.polynomial.polynomial.polyval(input_, w)[0] - output_
    return ((errors**2).sum())/(input_.shape)

def trainData(FileName, lamda): 
    input_train = np.genfromtxt(FileName + '.csv', dtype=float, delimiter=',')
    
    ones = np.ones(input_train.shape[0])
    ones = ones.reshape(ones.shape[0], 1)
    
    X = input_train[:, 0]
    X = X.reshape(X.shape[0], 1)
    X = np.concatenate((ones, X), axis=1)
    for i in range(2, degree+1):
        temp = (X[:,1])**i
        temp = temp.reshape(input_train.shape[0], 1)
        X = np.concatenate((X, temp), axis = 1)
    y = input_train[:, 1]
    y = y.reshape(y.shape[0], 1)
    
    Xtransp_X = np.dot((X.T),X)
    regularized_Xtransp_X_inv = np.linalg.inv(Xtransp_X + lamda*np.eye(X.shape[1]))
    weights = np.dot(np.dot(regularized_Xtransp_X_inv, X.T), y)
  
    return weights

def lamdaTest(input_train, output_train, input_valid, output_valid):
    lamda_range = np.linspace(0.0, 1.0, 20)
    minError = 100
    Train_errors = []
    Valid_errors = []
    
    
    for lamda in lamda_range:
        FileName = 'Dataset_1_train'
        
        weights = trainData(FileName, lamda)
        train_error = MSE(input_train, output_train, weights)
        valid_error = MSE(input_valid, output_valid, weights)
        Train_errors.append(train_error)
        Valid_errors.append(valid_error)
        
        if valid_error < minError:
            minError = valid_error
            bestLamda = lamda
        
    return ((Train_errors, Valid_errors), bestLamda)

############################   MAIN PROGRAM    ################################
#Calculate the weights for a range of lamda values to find the best one
#Plot taining error and validation error vs lamda to find ideal lambda value
#Best lamda value corresponds to the lowest validation error, and found to be 0.017
    
#lamda = lamdaTest(input_train, output_train, input_valid, output_valid)[1]
T_V_errors = lamdaTest(input_train, output_train, input_valid, output_valid)[0]

plt.plot(np.linspace(0.0, 1.0, 20), T_V_errors[0], 'r-')
plt.plot(np.linspace(0.0, 1.0, 20), T_V_errors[1], 'g-')
plt.xlabel('lamda')
plt.ylabel('errors')
plt.title('Lambda test')
plt.legend(['training error', 'validation error'], loc=4)
plt.axis([0.0, 1.0, 0.0, 12.0])
plt.show()

#Train the data using our best lamda value
FileName = 'Dataset_1_train'
w = trainData(FileName, lamda)

#Evaluate polynomial with weights w to find predicted output values
x_range = np.linspace(-1.0, 1.0, 250)
Y_predicted = np.polynomial.polynomial.polyval(x_range, w)[0]

#Calculate Mean Square Error
train_mse = MSE(input_train, output_train, w)
valid_mse = MSE(input_valid, output_valid, w)
test_mse = MSE(input_test, output_test, w)

############################   PRINT STATEMENTS   #############################
if print_on:
    print("Training MSE: " + str(train_mse[0]))
    print("Valid MSE: " + str(valid_mse[0]))
    print("Test MSE: " + str(test_mse[0]))

###########################   PLOTS    ########################################
if plot_on:
    
    #plot predicted curve along with data points - for training set
    plt.subplot(311)
    plt.plot(input_train, output_train, 'ro')
    plt.plot(x_range, Y_predicted, 'g-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression - Training set')
    plt.legend(['data points', 'predicted'], loc=2)
    plt.axis([-1.5, 1.5, -30.0, 45.0])
    
    #plot predicted curve along with data points - for validation set
    plt.subplot(312)
    plt.plot(input_valid, output_valid, 'ro')
    plt.plot(x_range, Y_predicted, 'g-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression - Validation set')
    plt.legend(['data points', 'predicted'], loc=2)
    plt.axis([-1.5, 1.5, -30.0, 45.0])

    #plot predicted curve along with data points - for test set
    plt.subplot(313)
    plt.plot(input_test, output_test, 'ro')
    plt.plot(x_range, Y_predicted, 'g-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression - Test set')
    plt.legend(['data points', 'predicted'], loc=2)
    plt.axis([-1.5, 1.5, -30.0, 45.0])
    
    plt.tight_layout()
    plt.show()

