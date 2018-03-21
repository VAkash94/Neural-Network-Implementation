"""
File name: executeMLP.py
Language: Python3
Author :
Akash Venkatachalam

"""

import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    """
    Function to obtain the sigmoid value for the given input
    :param x: The given input
    """
    denom = 1 + np.exp(-1*x)
    return 1/denom


def one_layer_forward(data_in, hidden_size,weight_1):
    """
    Function to perform feed-forward from one layer to the next layer
    :param data_in: Value in each record
    :param hidden_size: Size of the hidden layer
    :param weight_1: Weight matrix between the layers
    """

    assert (hidden_size == len(weight_1))  # Assertion to check if the hidden layer size equals the weight matrix size
    n = len(data_in)
    hidden_vals = [0]*hidden_size
    for i in range(hidden_size):
        ith_dot_prod = np.dot([1]+ data_in, weight_1[i])   # Performing dot product for input data and the weight matrix
        hidden_vals[i] = sigmoid(ith_dot_prod)
    return hidden_vals


def forward(data_in, hidden_size, weight_1, weight_2):
    """
    Function to perform the feed-forward propagation through both the layers.
    :param data_in: Value in each record
    :param hidden_size: Size of the hidden layer
    :param weight_1: Weight matrix of input to hidden layer
    :param weight_2: Weight matrix of hidden to output layer
    """
    hidden_vals = one_layer_forward(data_in, hidden_size, weight_1)   # Performing feed-forward from input to hidden layer
    out_size = len(weight_2)
    out_vals = one_layer_forward(hidden_vals, out_size, weight_2)     # Performing feed-forward from hidden to output layer
    return (hidden_vals, out_vals)


def classifier(in_data, w1, w2):
    """
    Function to perform feed forward and obtain the values
    :param in_data: Value in each record
    :param w1: Weight matrix of input to hidden layer
    :param w2: Weight matrix of hidden to output layer
    """
    hid_nodes = w1.shape[0]
    all_vals = forward(in_data, hid_nodes, w1, w2)
    predicted_vec = all_vals[1]
    index_max = np.argmax(predicted_vec)
    return int(index_max)+1


def confusion_matrix(file_name_test, w1, w2):
    """
    Function to evaluate the confusion matrix using the weights on testing dataset
    :param file_name_test: Name of the test dataset
    :param w1: The weights of input to hidden layer obtained from running training dataset
    :param w2: The weights of hidden to output layer obtained from running training dataset
    """
    test = np.array(pd.read_csv(file_name_test, header=None))
    no_labels = w2.shape[0]
    no_features = w1.shape[1] - 1                   # Excluding bias
    labels = list(range(1, no_labels + 1))
    conf_mat = np.zeros((no_labels, no_labels)).astype(int)   # Creating an empty confusion matrix with zeroes
    for i in range(len(test)):
        this_sample = list(test[i][:no_features])   # Classifier accepts lists
        predicted = classifier(this_sample, w1, w2)
        actual = int(test[i][-1])
        i = labels.index(predicted)
        j = labels.index(actual)
        conf_mat[i, j] += 1
    # Drawing confusion matrix using Seaborn
    df_cm = pd.DataFrame(conf_mat, index=[i for i in labels],   # Loading the confusion matrix into data frame
                         columns=[i for i in labels])
    plt.figure(figsize=(5, 3))
    sn.set(font_scale=1.4)          # for label size
    sn.heatmap(df_cm, annot=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Actual", fontsize=16)
    plt.ylabel("Predicted", fontsize=16)
    plt.show()
    plt.close()
    # End of drawing confusion matrix
    return conf_mat

def contour_plot(w1,w2):
    """
    Function to plot the decision boundaries for each epoch based on the weights.
    """

    x = np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)
    dim = len(x)
    #X, Y = np.meshgrid(x, y)

    Z = np.zeros((dim, dim))
    r,c = Z.shape
    for i in range(r):
        for j in range(c):
            Z[i,j] = classifier([x[i], y[j]], w1, w2)

    CS = plt.contourf(x, y, Z, 3)
    CS2 = plt.contour(CS, levels=CS.levels[::3])
    #cbar = plt.colorbar(CS)
    #cbar.add_lines(CS2)
    plt.title('Classification Region')
    plt.xlabel('Symmetry')
    plt.ylabel('Eccentricity')
    plt.show()



########################################################################################################################


profit_table= np.array([[20, -7, -7,-7],               # The given profit table represented in numpy array
                       [-7,15,-7, -7],
                       [-7, -7, 5, 7],
                       [-3, -3, -3, -3]])


w1 = np.array(pd.read_csv("W1.csv", header=None))      # Reading in the weights obtained by training set from stored files
w2 = np.array(pd.read_csv("W2.csv", header=None))


conf_mat = confusion_matrix("test_data.csv", w1, w2)   # Function call to evaluate confusion matrix
print("The accuracy of your model is ", int(np.sum(np.diagonal(conf_mat))/np.sum(conf_mat)*100), "%")
print("The total profit in cents for this model is:", np.sum(conf_mat*profit_table))
print("Your confusion matrix for this mdoel is: \n", conf_mat )

contour_plot(w1,w2)
