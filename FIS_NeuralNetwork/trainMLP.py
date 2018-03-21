"""
File name: trainMLP.py
Language: Python3
Author :
Akash Venkatachalam

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Function to obatin the sigmoid value for the given input
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


def one_layer_backward(data_in, hidden_vals, weight):
    """
    Function to perform one layer of back propagation
    :param data_in: Value in each record
    :param hidden_vals: Size of the hidden layer
    :param weight: The weight matrix
    """
    hidden_size = len(hidden_vals)
    assert (hidden_size == len(weight))
    delta_mid = [0]*hidden_size
    for i in range(hidden_size):
        dot_prod = np.dot(weight[i], data_in)       # Performing the dot product of values and weights
        delta_mid[i] = dot_prod * hidden_vals[i] * (1 - hidden_vals[i])
    return delta_mid


def deltas(in_data, hidden_size, w1, w2, correct_y):
    """
    Function to calculate the delta or error
    :param in_data: Value in each record
    :param hidden_size: Size of the hidden layer
    :param w1: Weight matrix between input and hidden layer
    :param w2: Weight matrix between hidden and output layer
    :param correct_y: Correctly classified instance obtained from training set
    :return:
    """
    out = forward(in_data, hidden_size, w1, w2)
    outer_deltas = (np.array(out[1]) - correct_y) * np.array(out[1]) * (1 - np.array(out[1]))
    deltas = one_layer_backward(outer_deltas, out[0], np.transpose(w2[:, 1:]))
    return (deltas, outer_deltas)


def one_layer_weight_updator(etha, weight, in_data, next_delta):
    """
    Function to update the weight of matrix for one layer
    :param etha: The value of learning rate, it is set to 0.01
    :param weight: The old weight matrix
    :param in_data: Value of each record
    """
    dim = weight.shape
    m = dim[1]
    n = dim[0]
    new_weight = weight - etha*np.outer(next_delta, in_data)    # Updating the weight matrix
    return new_weight


def weight_updator(row, hidden_size, w1, w2, correct_y_labels):
    """
    Function to perform the weight updating in back propagation
    :param row: For each row in the given dataset
    :param hidden_size: Size of the hidden layer
    :param w1: Weight matrix between input and hidden layer
    :param w2: Weight matrix between hidden and output layer
    :param correct_y_labels: Correctly classified instance obtained from training set
    """
    pair_deltas = deltas(row, hidden_size, w1, w2, np.array(correct_y_labels))
    out = forward(row, hidden_size, w1, w2)
    new_w2 = (one_layer_weight_updator(.01, w2, [1] + out[0], pair_deltas[1]))   # Updating the weights
    new_w1 = (one_layer_weight_updator(.01, w1, [1] + row, pair_deltas[0]))
    return (new_w1, new_w2)


def data_transformer(file_name):
    """
    Function to take in the label and turn it into a binary type, that would be given by the classifier.
    For instance, class 1 will have a binary transformation of 1000.
    :param file_name: Name of the file to perform this transformation.
    """
    df = pd.read_csv(file_name, header = None)
    dim = df.shape
    data = np.array(df)
    labels = list(map(str, list(set(data[:,2].astype(int)))))   # The label determines where 1 should be placed
    n = dim[0]
    new_train = []
    for i in range(n):
        loc = labels.index(str(int(data[i][-1])))   # Finding the index of that label
        binary_vec = [0,0,0,0]                      # Initializing a new binary value
        binary_vec[loc] = 1                         # Changing the value of 0 to 1 based on the class
        new_train.append(list(data[i][:-1]) + binary_vec)
    return (new_train, dim[1]-1, len(labels))


def learner_by_epoch(epoch):
    """
    Function to perform the feed-forward and backward propagation. It also calculates the SSE value
    :param epoch: The number of epochs to perform feed-forward and backward propagation
    """
    sse_list = []
    w1 = (2*np.random.rand(no_hidden_nodes, no_features + 1))-1  # Randomizing weights in the range [-1,+1]
    w2 = (2*np.random.rand(no_classes, no_hidden_nodes + 1))-1
    no_rows = len(input_binary_output)
    for iter in range(epoch):
        sse = 0
        for i in range(no_rows):
            all_vals = forward(input_binary_output[i][0:2], no_hidden_nodes, w1, w2)    # Calling feed-forward
            actual_y = np.array(input_binary_output[i][2:])
            out_vals = all_vals[1]
            sse += np.dot(out_vals - actual_y, out_vals - actual_y)     # Calculating the SSE value
            new_ws = weight_updator(input_binary_output[i][0:2], no_hidden_nodes, w1, w2, actual_y)
            w1, w2 = new_ws
        sse_list.append(sse)
    return (w1, w2, sse_list)



##############################################################################################################


triple = data_transformer("train_data.csv")     # Reading in the training file
input_binary_output = triple[0]
no_features = triple[1]
no_classes = triple[2]
no_hidden_nodes = 5

updates = learner_by_epoch(10)            # Updating weights for the given number of epochs
print("The W1 is \n ", updates[0], "\n")
print("The W2 is \n", updates[1], "\n")
plt.plot(updates[2], "b-")                  # Plotting SSE vs Epoch
plt.xlabel("The Epoch")                     # Labeling the x-axis of the plot
plt.ylabel("Sum of Squared Error (SSE)")    # Labeling the y-axis of the plot
plt.title("SSE vs epoch")                   # Labeling the title of the plot
plt.show()
plt.close()

np.savetxt('W1.csv', updates[0], delimiter=",")     # Saves the weights one layer in a CSV file
np.savetxt('W2.csv', updates[1], delimiter=",")     # Saves the weights of next layer in a CSV file
