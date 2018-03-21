Description:
The aim is to train and test both Multi-layer perceptron and the decision tree.

##############################################################################

How to Run:
* The code needs to be compiled using Python 3 interpreter.
* The code uses the following libraries:
	math library, numpy library, matplotlib.pyplot, pandas and seaborn libraries
* For MLP part, first trainMLP.py need to run in order to train the MLP classifier. It produces two weight files, one for each layer. Then executeMLP.py need to be run on the 
test dataset. It uses the stored weights obtained from training samples.
* For decision tree part, trainDT.py need to run in order to train and create rules for decision tree classifier. This tree is stored in pickle object. Then the executeDT.py need to 
run on test dataset which will use the rules obtained from training set.
* To change the input maze file, pass a different text file as the argument.

##############################################################################

Interpreting the output:
The output prints the weight values and the learning curve (SSE vs Epoch). For the test data set, it prints out the accuracy and confusion matrix.

##############################################################################
