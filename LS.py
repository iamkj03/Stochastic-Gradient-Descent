##############################################
#CSE5523 Machine Learning                    #
#HW2                                         #
#Stochastic Gradient Descent implementation  #
#Kyeong Joo Jung                             #
#500411516                                   #
##############################################

import argparse
import numpy as np

###Using argparse to set parameters
parser = argparse.ArgumentParser(description="Please enter the parameters(train_data.txt/train_target.txt/beta/output.txt/learning rate/maxiter/tol) needed")
parser.add_argument("-A", help="train data text file")
parser.add_argument("-y", help="train target text file")
parser.add_argument("-beta", type=float, help="beta value which is greater or equal to 0")
parser.add_argument("-x", help="output text file")
parser.add_argument("-lr", type=float, help="learning rate")
parser.add_argument("-maxiter", type=int, help="number of maximum iteration")
parser.add_argument("-tol", type=float, help="specifies the max |x_t+1 - x_t| allowed")
args = parser.parse_args()

### variables set from the parameters
train_data = str(args.A)
train_target = str(args.y)
beta_value = float(args.beta)
output_file = str(args.x)
learning_rate = float(args.lr)
max_iter = int(args.maxiter)
tol = float(args.tol)

### Stochastic gradient descent function - need train_data, train_target, learning rate, max iteration
def SGD(X_data, X_target, learning_rate, beta_value, max_iter):
    row, column = X_data.shape                      ##values of rows and columns of the data
    w = np.zeros(shape=(row, 1))                    ##initial weight value to 0s
    before_w = np.zeros(shape=(row, 1))             ##previous weight value
    min = [100]                                     ##variable to keep the smallest error
    final_w = []                                    ##variable to put the optimal weight value

    for a in range(max_iter):                       ##iterations up to maximum iterations

        w_gra = np.zeros(shape=(row, 1))            ##setting initial value of gradient of weight to 0s

        for i in range(column):                     ##calculation to get the weight variable value2
            pred = np.dot(X_data[i], w)
            target = X_target[i]
            error = 0.5 * np.linalg.norm(pred - target)**2 + 0.5 * beta_value * np.linalg.norm(w)**2        ##error calculation from HW2 1/2||Ax-y||**2+ beta/2||x||**2

            w_gra = X_data[i].reshape(row, 1) * (pred - target) + beta_value * w
            w = w - learning_rate * w_gra                                                                   ##calculation to get weight gradient and weight including the learning rate

            if i == 0:
                before_w = w                                                                                ##initial value of weight used for the tolerance value

            for b in range(row):
                if abs(before_w[b] - w[b]) > tol:
                    print("Stopped training due to tol value")                                              ##prints error if the difference goes above tol value
                    return final_w                                                                          ##comparing x_t+1 and x_t with tol value/ if the difference is over tol value, training ends and return final weight

            before_w = w                                                                                    ##setting x_t to x_t+1 for next iteration

            if min[0]>error:
                min[0] = error
                final_w = w                                                                                 ##calculation to get final weight and minimum error

    return final_w                                                                                          ##returning optimal weight

if __name__ == '__main__':
    print("train_data: " + train_data)
    print("train_target: " + train_target)
    print("beta_value: ", beta_value)
    print("output_file: " + output_file)
    print("learning_rate: ", learning_rate)
    print("max_iter: ", max_iter)
    print("tol: ", tol)                                                                                     ##printing out the parameters

    X_data = np.genfromtxt(train_data, delimiter=" ")
    X_target = np.genfromtxt(train_target)
    X_data = np.array(X_data)
    X_target = X_target.reshape(len(X_data), 1)                                                             ##preprocessing the data

    file = open(output_file, "w")
    for row in SGD(X_data, X_target, learning_rate, beta_value, max_iter):
        np.savetxt(file, row)                                                                               ##exporting the txt file with the values from SGD function

