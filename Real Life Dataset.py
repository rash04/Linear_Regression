"""
COMP 551 (Applied Machine Learning) Assignment 1 Question 3
"REAL LIFE DATASET"
Name: RASHIK HABIB
McGill University
Date: 26th January, 2018
"""

import numpy as np
import matplotlib.pyplot as plt

###################################   VARIABLES    ############################
print_on = 1
plot_on = 1
regularization_on = 1

degree = 1
examples = 1596
features = 101

####################################    FUNCTIONS    ##########################
#Mean Squared Error
def MSE(input_, output_, w):
    errors = output_ - np.dot(input_, w)
    return ((errors**2).sum())/(errors.shape[0])

#Finds the weights required given the training data set and lamdba value,
def trainData(FileNumber, lamda): 
    input_train = np.genfromtxt('communities/DatasetsCandC/CandC-train' + str(FileNumber) + '.csv', dtype=float, delimiter=',')
    
    ones = np.ones(input_train.shape[0])
    ones = ones.reshape(ones.shape[0], 1)
    
    X = input_train[:, :-1]
    X = np.concatenate((X, ones), axis=1)
    y = input_train[:, -1]
    y = y.reshape(y.shape[0], 1)
    
    Xtransp_X = np.dot((X.T),X)
    regularized_Xtransp_X_inv = np.linalg.inv(Xtransp_X + lamda*np.eye(X.shape[1]))
    weights = np.dot(np.dot(regularized_Xtransp_X_inv, X.T), y)
  
    return weights   

#Tests the weights provided a filenumber is given, by calculating the MSE between
#the actual value in the test data and the predicted output value
def testData(FileNumber, weights):
    input_test = np.genfromtxt('communities/DatasetsCandC/CandC-test' + str(FileNumber) + '.csv', dtype=float, delimiter=',')
    
    ones = np.ones(input_test.shape[0])
    ones = ones.reshape(ones.shape[0], 1)
    
    X = input_test[:, :-1]
    X = np.concatenate((X, ones), axis=1)
    y = input_test[:, -1]
    y = y.reshape(y.shape[0], 1)
    
    error = MSE(X, y, weights)
    return error

#Goes through a list of lamda values to find the one which returns the lowest error
#I started with a very large range, and subsequently narrowed to find the most
#accurate value for lamda
def lamdaTest():
    lamda_range = np.linspace(1.25, 1.27, 20)
    minError = 100
    
    for lamda in lamda_range:
        allErrors = []
        for i in range(1, 6):
            weights = trainData(i, lamda)
            error = testData(i, weights)
            allErrors.append(error)
            
        avgError = np.average(allErrors)
        
        if avgError < minError:
            minError = avgError
            bestLamda = lamda
    
    return bestLamda

    
###############################   MAIN PROGRAM    #############################
if regularization_on:
    #lamda = lamdaTest()
    lamda = 1.2689473684210526
else:
    lamda = 0.0

allErrors = []
allWeights = []

for i in range(1, 6):
    weights = trainData(i, lamda)
    error = testData(i, weights)
    print(error)
    
    allErrors.append(error)
    allWeights.append(weights)

avgError = np.average(allErrors)

print("5-fold cross validation error using lamda=" +str(lamda) + " is: " +str(avgError))

#The learnt parameters "weights" have been exported to a .csv file named .... 

#Since the minimum MSE is seen to occur in set 4, we find the selected features using this set
#and find the corresponding MSE for the reduced number of features
selected_features = [index for index, weight in enumerate(allWeights[3]) if abs(weight)>=0.10]
print("reduced number of features for set 4: " + str(len(selected_features)))

input_test = np.genfromtxt('communities/DatasetsCandC/CandC-test' + str(4) + '.csv', dtype=float, delimiter=',')

ones = np.ones(input_test.shape[0])
ones = ones.reshape(ones.shape[0], 1)

X = input_test[:, selected_features]

y = input_test[:, -1]
y = y.reshape(y.shape[0], 1)

reduced_weights = allWeights[3][selected_features, :]
error = MSE(X, y, reduced_weights)

#The best fit gives us an error of...
print("Reduced features gives error:" + str(error))





    
    



