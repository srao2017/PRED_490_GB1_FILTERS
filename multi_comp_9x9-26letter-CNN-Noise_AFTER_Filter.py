# -*- coding: utf-8 -*-
# We will randomly define initial values for connection weights, and also randomly select
#   which training data that we will use for a given run.

import random

# We want to use the exp function (e to the x); it's part of our transfer function definition
from math import exp

# Biting the bullet and starting to use NumPy for arrays
import numpy as np

# So we can make a separate list from an initial one

import os
import csv
import pandas as pd
import pickle as pk
import sys

# For pretty-printing the arrays
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True) 

###############################################################################
# create_outfile
# Creates an output file with filename containing NN parameters passed in
# Writes the first heading row and returns the file handle and writer
# Added by Sanjay
###############################################################################
def create_outfile(filename,hnodes,alpha,eta,epsi,iters, noiseLevel=0, cnn_mode=0):

    outfile_name = str(filename)+"_"+str(hnodes)+"_"+str(alpha)+"_"+str(eta)+"_"+str(epsi)+"_"+str(iters)+"_"+str(noiseLevel)+str(".csv")
    if cnn_mode > 0:
        outfile_name = str("CNN") + outfile_name
        
    outfile = open(outfile_name, 'wb')
    
    wr = csv.writer(outfile)
    outlist = ["Letter", "Tot. SSE"]

    for i in range(hnodes):
            outlist.append(str('H')+str(i))

    f = open(str(filename)+str("_classes.csv"),"rb")
    df = pd.read_csv(f)
    f.close()
    
    cl = pd.Series(df["Class"]).unique()
    cl.sort()
    for c in cl:
        letters = df[df['Class'] == c]
        v = letters["Letter"]
        vl = v.tolist()
        outlist.append(vl)
    
    wr.writerow(outlist)
    return outfile, wr
    
###############################################################################
# save_to_outfile
# Save results to the output file
# Letter, SSE, Hidden array values, Output array values, Input to Hidden weights etc.
# Added by Sanjay
###############################################################################
def save_to_outfile(wr, letter, sse, H_array, O_array, W_array, Letter_grid, cnn_mode):
        # write the output
        outlist = [letter, sse]

        for i in range(len(H_array)):
            outlist.append(H_array[i])
            
        for i in range(len(O_array)):
            outlist.append(O_array[i])
           
        if cnn_mode == 0:
            for i in range(len(W_array)):
                outlist.append(np.reshape(W_array[i],(9,9)))
            
        wr.writerow(outlist)
        
       
###############################################################################
# close_outfile
# Close the output file
# Added by Sanjay
###############################################################################
def close_outfile(outfile, wr, h_nodes, iterations, alpha, eta, epsilon,LetterIters):
    outlist=["Hidden Nodes", h_nodes]
    wr.writerow(outlist)
    outlist=["Iterations", iterations]
    wr.writerow(outlist)
    outlist=["alpha", alpha]
    wr.writerow(outlist)
    outlist=["eta", eta]
    wr.writerow(outlist)
    outlist=["epsilon", epsilon]
    wr.writerow(outlist)
    outlist=["Letter Iterations",LetterIters]
    wr.writerow(outlist)
    outfile.close()
    


####################################################################################################
#
# Procedure to welcome the user and identify the code
#
####################################################################################################

def welcome ():


    print
    print '******************************************************************************'
    print
    print 'Welcome to the Multilayer Perceptron Neural Network'
    print '  trained using the backpropagation method.'
    print 'Version 0.2, 01/22/2017, A.J. Maren'
    print 'For comments, questions, or bug-fixes, contact: alianna.maren@northwestern.edu'
    print ' ' 
    print 'This program learns to distinguish between five capital letters: X, M, H, A, and N'
    print 'It allows users to examine the hidden weights to identify learned features'
    print
    print '******************************************************************************'
    print
    return()

        

####################################################################################################
####################################################################################################
#
# A collection of worker-functions, designed to do specific small tasks
#
####################################################################################################
####################################################################################################

   
#------------------------------------------------------#    

# Compute neuron activation using sigmoid transfer function
def computeTransferFnctn(summedNeuronInput, alpha):
    activation = 1.0 / (1.0 + exp(-alpha*summedNeuronInput)) 
    return activation
  

#------------------------------------------------------# 
    
# Compute derivative of transfer function
def computeTransferFnctnDeriv(NeuronOutput, alpha):
    return alpha*NeuronOutput*(1.0 -NeuronOutput)     


#------------------------------------------------------# 
def matrixDotProduct (matrx1,matrx2):
    dotProduct = np.dot(matrx1,matrx2)
    return(dotProduct)    


####################################################################################################
####################################################################################################
#
# Function to obtain the neural network size specifications
#
####################################################################################################
####################################################################################################

def obtainNeuralNetworkSizeSpecs (numInputNodes = 81, outputNodes=26):

# This procedure operates as a function, as it returns a single value (which really is a list of 
#    three values). It is called directly from 'main.'
#        
# This procedure allows the user to specify the size of the input (I), hidden (H), 
#    and output (O) layers.  
# These values will be stored in a list, the arraySizeList. 
# This list will be used to specify the sizes of two different weight arrays:
#   - wWeights; the Input-to-Hidden array, and
#   - vWeights; the Hidden-to-Output array. 
# However, even though we're calling this procedure, we will still hard-code the array sizes for now.   

    numHiddenNodes = input("Enter number of hidden nodes: ")
    numOutputNodes = outputNodes   
    print ' '
    print '  The number of nodes at each level are:'
    print '    Input: 9x9 (square array)'
    print '    Hidden: ' + str(numHiddenNodes)
    print '    Output classes: ' + str(outputNodes)
            
# We create a list containing the crucial SIZES for the connection weight arrays                
    arraySizeList = (numInputNodes, numHiddenNodes, numOutputNodes)
    
# We return this list to the calling procedure, 'main'.       
    return (arraySizeList)  


####################################################################################################
#
# Function to initialize a specific connection weight with a randomly-generated number between 0 & 1
#
####################################################################################################

def InitializeWeight ():

    randomNum = random.random()
    weight=1-2*randomNum
    return (weight)  



####################################################################################################
####################################################################################################
#
# Function to initialize the node-to-node connection weight arrays
#
####################################################################################################
####################################################################################################

def initializeWeightArray (weightArraySizeList):

# This procedure is also called directly from 'main.'
#        
# This procedure takes in the two parameters, the number of nodes on the bottom (of any two layers), 
#   and the number of nodes in the layer just above it. 
#   It will use these two sizes to create a weight array.
# The weights will initially be assigned random values here, and 
#   this array is passed back to the 'main' procedure. 

    
    numBottomNodes = weightArraySizeList[0]
    numUpperNodes = weightArraySizeList[1]

# Initialize the weight variables with random weights    
    weightArray = np.zeros((numUpperNodes,numBottomNodes))    # iniitalize the weight matrix with 0's
    for row in range(numUpperNodes):  #  Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input matrix (expressed as a column)
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes.
        for col in range(numBottomNodes):  # number of columns in matrix 2
            weightArray[row,col] = InitializeWeight ()
                  
    
# We return the array to the calling procedure, 'main'.       
    return (weightArray)  


####################################################################################################
####################################################################################################
#
# Function to initialize the bias weight arrays
#
####################################################################################################
####################################################################################################

def initializeBiasWeightArray (numBiasNodes):

# This procedure is also called directly from 'main.'

# Initialize the bias weight variables with random weights    
    biasWeightArray = np.zeros(numBiasNodes)    # iniitalize the weight matrix with 0's
    for node in range(numBiasNodes):  #  Number of nodes in bias weight set
        biasWeightArray[node] = InitializeWeight ()
                  
# Print the entire weights array. 
    print biasWeightArray
                  
    
# We return the array to the calling procedure, 'main'.       
    return (biasWeightArray)  

###############################################################################
#
# Begin: 
#Functions to convert CSV specification to digitized letters
# and generate a text output file to be read into a training data set
#
###############################################################################

###############################################################################
#
# get_letters
# gets a specified letter from the filename (.CSV file) and colapses the pattern
# into a 1d array and returns the pattern
#
# Added by Sanjay
#
###############################################################################
def get_letters (filename, letter, noiseLevel):
    pattern = []
    i = 10
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            #find the letter
            if row[0] in letter:
                print(row[0])
                i = 0;
                continue
            #retrieve the pattern and collapse into one row
            if i < 9:
                pattern = pattern+row
                i+=1

    # Add noise to the training data here
    # pattern is 81 elements plus 2 elements for the class and letter which we 
    # subtract
    if noiseLevel > 0:
        for i in range(len(pattern)): 
            randNum =  random.randint(1,100)
            if randNum < noiseLevel: # Introduce noise
                if pattern[i] == '0': 
                    pattern[i] = '1'
                else: 
                    pattern[i] = '0'
                
    return pattern

###############################################################################
#
# put_letters
# writes a pattern for a letter and includes the letter class id and the letter
# as the 82nd and 83rd items (array index 81 and 82)
#
# Added by Sanjay
#
###############################################################################
def put_letters (f, pattern, letter_id, letter):

    writer = csv.writer(f)
    writer.writerow(pattern+list([letter_id,str("'")+letter+str("'")]))

###############################################################################
#
# digitizeCSVpattern
# Takes a filename of the .CSV file - do not specify the .CSV as this
# functions appends a .CSV to the filename suplied
#
# reads the letters specified from the CSV and converts them to an 1d array
# as required by the NN code and writes it out to an intermediate file named
# filename_out.txt (the _out.txt is added to filename by this code)
#
# Invokes the helper functions put_letters and get_letters
#
#
# Added by Sanjay
#
#
###############################################################################

def digitizeCSVpattern(filename, classification="GROUPS"):
    
    # This noise level is used if you want to introduce noise
    # into the training data one time before the training.
    # To be effective however, noise must be introduced in "main" within the while loop
    # of iterative training so that there is different "noise" for each
    # training sample. This noise is set to 0 because we introduce noise later
    # in main as mentioned above.
    #
    all_letters_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,
    'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,
    'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25}
    
    noiseLevel = 0 
    
    if classification == "ALL_26":
        f = open(filename+str("_classes_26.csv"),"rb")
    elif classification == "GROUPS":
        f = open(filename+str("_classes.csv"),"rb")
    else:
        print "Error resolving classes"
        sys.exit()
        
    df = pd.read_csv(f)
    all_letters_dict = pd.Series(df.Class.values,index=df.Letter).to_dict()
    f.close()
    numOutputClasses = df.Class.max() 
    
    letter = []
    with open(str(filename)+str(0)+str(".csv")) as f:
        reader = csv.reader(f)
        for row in reader:
            #find the letter
            if row[0] in all_letters_dict.keys():
                letter.append(row[0])
    
    letters = list(letter)

    #
    # Number of variant specification files - must be named as follows:
    # "filename"_0, "file_name"_1,... starting at 0 and ending at numVariantFiles -1
    # in case of numVariantFiles = 3, the file names must be
    # filename_0.csv, filename_1.csv, filename_2.csv
    #
    numVariantFiles = 3
    
    f = open(filename+str("_out.txt"),"wb")
    letter_id = 0
    for letter in letters:
        for vari in range(numVariantFiles):
            pattern = get_letters(str(filename)+str(vari)+str(".csv"),[letter], noiseLevel)
            if pattern is not None:
                #letter_id = ord(letter) - ord(letters[0])
                put_letters(f,pattern, all_letters_dict[letter], letter)
                letter_id = letter_id + 1
        
    f.close()
    cols = len(pattern)
    rows = len(letters)*numVariantFiles
    return rows, cols+2, numOutputClasses+1 
    #the +2 is for the classID and letterId that were added by put_letters

    
###############################################################################
# readTrainingData
# reads the training data from the CSV file specifying the letter patterns
# invokes the helper function digitizeCSVpattern
#
# Added by Sanjay
#
###############################################################################

def readTrainingData(filename, classification="GROUPS"):

    rows, cols, numOutputClasses = digitizeCSVpattern(filename, classification)
    
    trainingData =[]

    for row in xrange(rows): trainingData += [[0]*cols]
    i = 0
    fname = filename+str("_out.txt")
    with open(fname) as f:
        reader = csv.reader(f)
        for row in reader:
            trainingData[i] = list(row)
            len_pat = len(trainingData[i]) - 1
            for j in range(len_pat):
                trainingData[i][j] = int(trainingData[i][j])
            i += 1
    return trainingData, rows, numOutputClasses   


###############################################################################
# def maskedTrainingData
# Apply masks on the training data and create a new set of training data from
# the training input set
#  
# 9x9 becomes 4 times 9x9 as a 4 masks are applied in sequence
#
# Modified by Sanjay
#
###############################################################################

def maskedTrainingData (trainingData, mask):
    
    len_training_data = len(trainingData)
    len_training_data_letter = len(trainingData[0])
    maskedTrainingDataAll=[[]]*len_training_data
    #default masks
    mask_ids = [0,1,2,3]
    # get mask ids from user
    mask_ids = [int(x) for x in raw_input("Enter mask ids separated by space: ").split()]        

    for i in range(len_training_data):
        maskedTrainingData = []
        # only modify the letter pattern data
        letterTrainingData = np.array(trainingData[i][0:len_training_data_letter - 2])
        # retain to reconstruct at the end
        class_and_letter = trainingData[i][len_training_data_letter - 2:]
        for mask_num in mask_ids:            
            maskedTrainingData_temp = applyMask (letterTrainingData, len(letterTrainingData), mask, mask_num, 0, 9, 9)
            maskedTrainingData = np.append(maskedTrainingData,maskedTrainingData_temp)
            
        # insert the last two elements class and letter
        maskedTrainingData = np.append(maskedTrainingData, class_and_letter)
            
        maskedTrainingDataAll[i] = maskedTrainingData[:]
        
    return maskedTrainingDataAll, len(maskedTrainingData)
   

###############################################################################
# End: 
# Functions to convert CSV specification to digitized letters
# and generate a text output file to be read into a training data set
###############################################################################

###############################################################################
###############################################################################
#
# Function to return a trainingDataList
#
###############################################################################
###############################################################################

def obtainSelectedAlphabetTrainingValues (trainingDataSetNum, trainingData):
    
    if trainingDataSetNum >= 0 and trainingDataSetNum < len(trainingData):   
        return trainingData[trainingDataSetNum]    
    else:
        return trainingData[0]


###############################################################################
###############################################################################
#
# Function to initialize a specific connection weight with a randomly-generated number between 0 & 1
#
###############################################################################
###############################################################################

def obtainRandomAlphabetTrainingValues (trainingData, numTrainingDataSets):

   
    # The training data list will have 11 values for the X-OR problem:
    #   - First nine valuea will be the 5x5 pixel-grid representation of the letter
    #       represented as a 1-D array (0 or 1 for each)
    #   - Tenth value will be the output class (0 .. totalClasses - 1)
    #   - Eleventh value will be the string associated with that class, e.g., 'X'
    # We are starting with five letters in the training set: X, M, N, H, and A
    # Thus there are five choices for training data, which we'll select on random basis
     
    trainingDataSetNum = random.randint(0, numTrainingDataSets-1)
    return (trainingData[trainingDataSetNum])  

            
###############################################################################
###############################################################################
#
# Perform a single feedforward pass
#
###############################################################################
###############################################################################



###############################################################################
#
# Function to initialize a specific connection weight with a randomly-generated
# number between 0 & 1
#
###############################################################################


def ComputeSingleFeedforwardPassFirstStep (alpha, arraySizeList, inputDataList, wWeightArray, 
biasHiddenWeightArray):

    hiddenArrayLength = arraySizeList [1]

    # iniitalize the sum of inputs into the hidden array with 0's  
    sumIntoHiddenArray = np.zeros(hiddenArrayLength)    
    hiddenArray = np.zeros(hiddenArrayLength)   

    sumIntoHiddenArray = matrixDotProduct (wWeightArray,inputDataList)
    
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        hiddenNodeSumInput=sumIntoHiddenArray[node]+biasHiddenWeightArray[node]
        hiddenArray[node] = computeTransferFnctn(hiddenNodeSumInput, alpha)

                                                                                                
    return (hiddenArray);
  


###############################################################################
#
# Function to compute the output node activations, given the hidden node 
# activations, the hidden-to output connection weights, 
# and the output bias weights.
# Function returns the array of output node activations.
#
###############################################################################

def ComputeSingleFeedforwardPassSecondStep (alpha, arraySizeList, hiddenArray, vWeightArray, 
biasOutputWeightArray):

    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]
    
# iniitalize the sum of inputs into the hidden array with 0's  
    sumIntoOutputArray = np.zeros(hiddenArrayLength)    
    outputArray = np.zeros(outputArrayLength)   

    sumIntoOutputArray = matrixDotProduct (vWeightArray,hiddenArray)
    
    for node in range(outputArrayLength):  #  Number of hidden nodes
        outputNodeSumInput=sumIntoOutputArray[node]+biasOutputWeightArray[node]
        outputArray[node] = computeTransferFnctn(outputNodeSumInput, alpha)
                                                                                                   
    return (outputArray);
  


###############################################################################
#
# Procedure to compute the output node activations and determine errors across
# the entire training data set, and print results.
#
###############################################################################

def ComputeOutputsAcrossAllTrainingData (alpha, arraySizeList, numTrainingDataSets, wWeightArray, 
biasHiddenWeightArray, vWeightArray, biasOutputWeightArray, trainingData, training=1, writer=None, cnn_mode=0):

    selectedTrainingDataSet = 0
    inputArrayLength = arraySizeList [0]
    outputArrayLength = arraySizeList [2]                               
    

    while selectedTrainingDataSet < numTrainingDataSets: 
        trainingDataList = obtainSelectedAlphabetTrainingValues (selectedTrainingDataSet, trainingData)


        inputDataList = [] 
        for node in range(inputArrayLength): 
            trainingDataRow = float(trainingDataList[node])  
            inputDataList.append(trainingDataRow)

        letterIdx = len(trainingDataList) - 1
        classIdx = len(trainingDataList) - 2

        hiddenArray = ComputeSingleFeedforwardPassFirstStep (alpha, arraySizeList, inputDataList, wWeightArray, biasHiddenWeightArray)
        outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, arraySizeList, hiddenArray, vWeightArray, biasOutputWeightArray)
        desiredOutputArray = np.zeros(outputArrayLength)    # iniitalize the output array with 0's
        desiredClass = int(trainingDataList[classIdx])      # identify the desired class
        desiredOutputArray[desiredClass] = 1                # set the desired output for that class to 1


                
        # Determine the error between actual and desired outputs
        #
        # Initialize the error array
        errorArray = np.zeros(outputArrayLength) 
    
        newSSE = 0.0
        #  Number of nodes in output set (classes)
        for node in range(outputArrayLength):  
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node]*errorArray[node]        

         
        selectedTrainingDataSet = selectedTrainingDataSet +1 
        if writer != None:
            save_to_outfile (writer, trainingDataList[letterIdx], newSSE, hiddenArray, outputArray, wWeightArray, inputDataList, cnn_mode)

                        

###############################################################################
#******************************************************************************
###############################################################################
#
#   Backpropgation Section
#
###############################################################################
#******************************************************************************
###############################################################################

   
            
####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the hidden-to-output connection weights
#
####################################################################################################
####################################################################################################


def BackpropagateOutputToHidden (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray, vWeightArray):

# The first step here applies a backpropagation-based weight change to the hidden-to-output wts v. 
# Core equation for the first part of backpropagation: 
# d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
# where:
# -- SSE = sum of squared errors, and only the error associated with a given output node counts
# -- v(h,o) is the connection weight v between the hidden node h and the output node o
# -- alpha is the scaling term within the transfer function, often set to 1
# ---- (this is included in transfFuncDeriv) 
# -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
# -- F = transfer function, here using the sigmoid transfer function
# -- Hidden(h) = the output of hidden node h. 

# We will DECREMENT the connection weight v by a small amount proportional to the derivative eqn
#   of the SSE w/r/t the weight v. 
# This means, since there is a minus sign in that derivative, that we will add a small amount. 
# (Decrementing is -, applied to a (-), which yields a positive.)

# For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand), 
#   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X. 
#   (Meaning: exact chapter is still TBD.) 
# For the latest updates, etc., please visit: www.aliannajmaren.com


# Unpack array lengths
    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]

    transferFuncDerivArray = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
 
                       
    deltaVWtArray = np.zeros((outputArrayLength, hiddenArrayLength))  # initialize an array for the deltas
    newVWeightArray = np.zeros((outputArrayLength, hiddenArrayLength)) # initialize an array for the new hidden weights
        
    for row in range(outputArrayLength):  #  Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes,
        #    and the columns correspond to the number of hidden nodes,
        #    which can be multiplied by the hidden node array (expressed as a column).
        for col in range(hiddenArrayLength):  # number of columns in weightMatrix
            partialSSE_w_V_Wt = -errorArray[row]*transferFuncDerivArray[row]*hiddenArray[col]
            deltaVWtArray[row,col] = -eta*partialSSE_w_V_Wt
            newVWeightArray[row,col] = vWeightArray[row,col] + deltaVWtArray[row,col]                                                                                     

                                                                                                                                                                                                           
    return (newVWeightArray);     

            
####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the bias-to-output connection weights
#
####################################################################################################
####################################################################################################


def BackpropagateBiasOutputWeights (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray):

# The first step here applies a backpropagation-based weight change to the hidden-to-output wts v. 
# Core equation for the first part of backpropagation: 
# d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
# where:
# -- SSE = sum of squared errors, and only the error associated with a given output node counts
# -- v(h,o) is the connection weight v between the hidden node h and the output node o
# -- alpha is the scaling term within the transfer function, often set to 1
# ---- (this is included in transfFuncDeriv) 
# -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
# -- F = transfer function, here using the sigmoid transfer function
# -- Hidden(h) = the output of hidden node h. 

# Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n, 
#   scales amount of change to connection weight

# We will DECREMENT the connection weight biasOutput by a small amount proportional to the derivative eqn
#   of the SSE w/r/t the weight biasOutput(o). 
# This means, since there is a minus sign in that derivative, that we will add a small amount. 
# (Decrementing is -, applied to a (-), which yields a positive.)

# For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand), 
#   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X. 
#   (Meaning: exact chapter is still TBD.) 
# For the latest updates, etc., please visit: www.aliannajmaren.com


# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in these equations 

# The equation for the actual dependence of the Summed Squared Error on a given bias-to-output 
#   weight biasOutput(o) is:
#   partial(SSE)/partial(biasOutput(o)) = -alpha*E(o)*F(o)*[1-F(o)]*1, as '1' is the input from the bias.
# The transfer function derivative (transFuncDeriv) returned from computeTransferFnctnDeriv is given as:
#   transFuncDeriv =  alpha*NeuronOutput*(1.0 -NeuronOutput), as with the hidden-to-output weights.
# Therefore, we can write the equation for the partial(SSE)/partial(biasOutput(o)) as
#   partial(SSE)/partial(biasOutput(o)) = E(o)*transFuncDeriv
#   The parameter alpha is included in transFuncDeriv


# Unpack the output array length
    outputArrayLength = arraySizeList [2]

    deltaBiasOutputArray = np.zeros(outputArrayLength)  # initialize an array for the deltas
    newBiasOutputWeightArray = np.zeros(outputArrayLength) # initialize an array for the new output bias weights
    transferFuncDerivArray = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
 

    for node in range(outputArrayLength):  #  Number of nodes in output array (same as number of output bias nodes)    
        partialSSE_w_BiasOutput = -errorArray[node]*transferFuncDerivArray[node]
        deltaBiasOutputArray[node] = -eta*partialSSE_w_BiasOutput  
        newBiasOutputWeightArray[node] =  biasOutputWeightArray[node] + deltaBiasOutputArray[node]           
   

                                                                                                                                                
    return (newBiasOutputWeightArray);     


####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the input-to-hidden connection weights
#
####################################################################################################
####################################################################################################


def BackpropagateHiddenToInput (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
    inputArray, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray):

# The first step here applies a backpropagation-based weight change to the input-to-hidden wts w. 
# Core equation for the second part of backpropagation: 
# d(SSE)/dw(i,h) = -eta*alpha*F(h)(1-F(h))*Input(i)*sum(v(h,o)*Error(o))
# where:
# -- SSE = sum of squared errors, and only the error associated with a given output node counts
# -- w(i,h) is the connection weight w between the input node i and the hidden node h
# -- v(h,o) is the connection weight v between the hidden node h and the output node o
# -- alpha is the scaling term within the transfer function, often set to 1 
# ---- (this is included in transfFuncDeriv) 
# -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
# -- F = transfer function, here using the sigmoid transfer function
# ---- NOTE: in this second step, the transfer function is applied to the output of the hidden node,
# ------ so that F = F(h)
# -- Hidden(h) = the output of hidden node h (used in computing the derivative of the transfer function). 
# -- Input(i) = the input at node i.

# Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n, 
#   scales amount of change to connection weight

# Unpack the errorList and the vWeightArray

# We will DECREMENT the connection weight v by a small amount proportional to the derivative eqn
#   of the SSE w/r/t the weight w. 
# This means, since there is a minus sign in that derivative, that we will add a small amount. 
# (Decrementing is -, applied to a (-), which yields a positive.)

# For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand), 
#   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X. 
#   (Meaning: exact chapter is still TBD.) 
# For the latest updates, etc., please visit: www.aliannajmaren.com

# Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n, 
#   scales amount of change to connection weight
 
# For the second step in backpropagation (computing deltas on the input-to-hidden weights)
#   we need the transfer function derivative is applied to the output at the hidden node        

# Unpack array lengths
    inputArrayLength = arraySizeList [0]
    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]              
                                          
# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in these equations       
    transferFuncDerivHiddenArray = np.zeros(hiddenArrayLength)    # initialize an array for the transfer function deriv 
      
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray[node]=computeTransferFnctnDeriv(hiddenArray[node], alpha)
        
    errorTimesTFuncDerivOutputArray = np.zeros(outputArrayLength) # initialize array
    transferFuncDerivOutputArray    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray              = np.zeros(hiddenArrayLength) # initialize array
      
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha)
        errorTimesTFuncDerivOutputArray[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray[outputNode]
        
    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray[hiddenNode] = weightedErrorArray[hiddenNode] \
            + vWeightArray[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray[outputNode]
             
    deltaWWtArray = np.zeros((hiddenArrayLength, inputArrayLength))  # initialize an array for the deltas
    newWWeightArray = np.zeros((hiddenArrayLength, inputArrayLength)) # initialize an array for the new input-to-hidden weights
        
    for row in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).

        for col in range(inputArrayLength):  # number of columns in weightMatrix
            partialSSE_w_W_Wts = -transferFuncDerivHiddenArray[row]*inputArray[col]*weightedErrorArray[row]
            deltaWWtArray[row,col] = -eta*partialSSE_w_W_Wts
            newWWeightArray[row,col] = wWeightArray[row,col] + deltaWWtArray[row,col]                                                                                     
                                                                  
    return (newWWeightArray);     
    
            
####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the bias-to-hidden connection weights
#
####################################################################################################
####################################################################################################


def BackpropagateBiasHiddenWeights (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
    inputArray, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray):

# The first step here applies a backpropagation-based weight change to the hidden-to-output wts v. 
# Core equation for the first part of backpropagation: 
# d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
# where:
# -- SSE = sum of squared errors, and only the error associated with a given output node counts
# -- v(h,o) is the connection weight v between the hidden node h and the output node o
# -- alpha is the scaling term within the transfer function, often set to 1
# ---- (this is included in transfFuncDeriv) 
# -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
# -- F = transfer function, here using the sigmoid transfer function
# -- Hidden(h) = the output of hidden node h. 

# Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n, 
#   scales amount of change to connection weight

# We will DECREMENT the connection weight biasOutput by a small amount proportional to the derivative eqn
#   of the SSE w/r/t the weight biasOutput(o). 
# This means, since there is a minus sign in that derivative, that we will add a small amount. 
# (Decrementing is -, applied to a (-), which yields a positive.)

# For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand), 
#   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X. 
#   (Meaning: exact chapter is still TBD.) 
# For the latest updates, etc., please visit: www.aliannajmaren.com


# Unpack array lengths
    inputArrayLength = arraySizeList [0]
    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]              
                                          
   
# Compute the transfer function derivatives as a function of the output nodes.
# Note: As this is being done after the call to the backpropagation on the hidden-to-output weights,
#   the transfer function derivative computed there could have been used here; the calculations are
#   being redone here only to maintain module independence              

    errorTimesTFuncDerivOutputArray = np.zeros(outputArrayLength) # initialize array    
    transferFuncDerivOutputArray    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray              = np.zeros(hiddenArrayLength) # initialize array    

    transferFuncDerivHiddenArray = np.zeros(hiddenArrayLength)  # initialize an array for the transfer function deriv 
    partialSSE_w_BiasHidden      = np.zeros(hiddenArrayLength)  # initialize an array for the partial derivative of the SSE
    deltaBiasHiddenArray         = np.zeros(hiddenArrayLength)  # initialize an array for the deltas
    newBiasHiddenWeightArray     = np.zeros(hiddenArrayLength)  # initialize an array for the new hidden bias weights
          
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray[node]=computeTransferFnctnDeriv(hiddenArray[node], alpha)      
                  
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha) 
        errorTimesTFuncDerivOutputArray[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray[outputNode]

    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray[hiddenNode] = weightedErrorArray[hiddenNode]
            + vWeightArray[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray[outputNode]
            
# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in these equations 


# ===>>> AJM needs to double-check these equations in the comments area
# ===>>> The code should be fine. 
# The equation for the actual dependence of the Summed Squared Error on a given bias-to-output 
#   weight biasOutput(o) is:
#   partial(SSE)/partial(biasOutput(o)) = -alpha*E(o)*F(o)*[1-F(o)]*1, as '1' is the input from the bias.
# The transfer function derivative (transFuncDeriv) returned from computeTransferFnctnDeriv is given as:
#   transFuncDeriv =  alpha*NeuronOutput*(1.0 -NeuronOutput), as with the hidden-to-output weights.
# Therefore, we can write the equation for the partial(SSE)/partial(biasOutput(o)) as
#   partial(SSE)/partial(biasOutput(o)) = E(o)*transFuncDeriv
#   The parameter alpha is included in transFuncDeriv

    for hiddenNode in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix           
        partialSSE_w_BiasHidden[hiddenNode] = -transferFuncDerivHiddenArray[hiddenNode]*weightedErrorArray[hiddenNode]
        deltaBiasHiddenArray[hiddenNode] = -eta*partialSSE_w_BiasHidden[hiddenNode]
        newBiasHiddenWeightArray[hiddenNode] = biasHiddenWeightArray[hiddenNode] + deltaBiasHiddenArray[hiddenNode]                                                                                                                                                                                                                                                         
  
                                                                                                                                            
    return (newBiasHiddenWeightArray); 

 
###############################################################################
# ComputeOutputsAcrossAllTrainedData
#
#
###############################################################################
  
def ComputeOutputsAcrossAllTrainedData (filename, arraySizeList, wWeightArray, 
    biasHiddenWeightArray, vWeightArray, biasOutputWeightArray):
    
    alpha =1
    fname = filename+str("_trained_out.csv")
    f = open(fname, "wb")
    wr = csv.writer(f)
    outlist = list()
    trainingData, numTrainingDataSets, outputNodes = readTrainingData(filename)

        
    for letter_num in range(len(trainingData)):
        inputDataList = trainingData[letter_num]
        inputDataList = inputDataList[:-2]
        hiddenArray = ComputeSingleFeedforwardPassFirstStep (alpha, arraySizeList, inputDataList, wWeightArray, biasHiddenWeightArray)
        outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, arraySizeList, hiddenArray, vWeightArray, biasOutputWeightArray)

        wr.writerow(outputArray)
        outlist.append (outputArray)
    f.close()
    print "Trained network output was written to: " + str(fname)
    return outlist
    
###############################################################################
# runTrainedNetwork
#
#
#
###############################################################################
def runTrainedNetwork():

    #
    # Use pickle to write to output files and store parameters
    #
    f = open("arraySizeList.pkl.txt", 'rb')
    arraySizeList = pk.load(f)
    f.close()

    f = open("wWeightArray.pkl.txt", 'rb')
    wWeightArray = pk.load(f)
    f.close()
    
    f = open("biasHiddenWeightArray.pkl.txt", 'rb')
    biasHiddenWeightArray = pk.load(f)
    f.close()
    
    f = open("vWeightArray.pkl.txt", 'rb')
    vWeightArray = pk.load(f)
    f.close()
    
    f = open("biasOutputWeightArray.pkl.txt", 'rb')
    biasOutputWeightArray = pk.load(f)
    f.close()

    filename = "ABCD_26_VARS"
    
    return ComputeOutputsAcrossAllTrainedData (filename, arraySizeList, wWeightArray, 
        biasHiddenWeightArray, vWeightArray, biasOutputWeightArray) 



####################################################################################################
####################################################################################################
#
# The following modules expand the boundaries around a chosen letter, and apply a masking filter to 
#   that expanded letter. The result is an array (9x9 in this case) of units, with activation values
#   where 0 <= v <= 1.  
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
# 
####################################################################################################
####################################################################################################
#
# Function to expand the grid containing a letter by one pixel in each direction
#
####################################################################################################
####################################################################################################
   
def expandLetterBoundaries (trainingDataList, eGH, eGW, len_gb1_classes):  

    expandedLetterArray = np.zeros(shape=(eGH,eGW)) 
    range_row = eGH-2
    range_col = eGW-2
    for i in range(range_row):#1 row padded on top and one padded on bottom
        for j in range(range_col): # 1 col padded to left and one padded to right
            expandedLetterArray[i+1][j+1] = trainingDataList[len_gb1_classes + i*(range_row)+j]
        
   
    return expandedLetterArray

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
###############################################################################
#
# Apply filters defined here
# Returns the arry of masks that may be used to filter the input
###############################################################################
    
def masks():
    mask = np.zeros(shape=(4,9))
    mask[0]  = [0,1,0,0,1,0,0,1,0] # vertical line middle
    mask[1]  = [0,0,0,1,1,1,0,0,0] # horizontal middle
    mask[2]  = [1,0,0,0,1,0,0,0,1] # diagonal 1
    mask[3]  = [0,0,1,0,1,0,1,0,0] # diagonal 2
    gridSize = int(len(mask[0])**0.5)
    return mask, gridSize, gridSize

    
###############################################################################
###############################################################################
#
# Function to return the letterArray after mask has been applied to it
#
###############################################################################
###############################################################################
    

def maskLetterFunc(expandedLetterArray, mask, gridHeight, gridWidth):

    
    maskLetterArray = np.zeros(shape=(gridHeight,gridWidth))
    
   
    rowVal = 1
    colVal = 1

    while rowVal <gridHeight+1: 

        arrayRow = rowVal - 1 

        while colVal <gridWidth+1:           
            e0 =  expandedLetterArray[rowVal-1, colVal-1]
            e1 =  expandedLetterArray[rowVal-1, colVal]
            e2 =  expandedLetterArray[rowVal-1, colVal+1]   
            e3 =  expandedLetterArray[rowVal, colVal-1]
            e4 =  expandedLetterArray[rowVal, colVal]
            e5 =  expandedLetterArray[rowVal, colVal+1]   
            e6 =  expandedLetterArray[rowVal+1, colVal-1]
            e7 =  expandedLetterArray[rowVal+1, colVal]
            e8 =  expandedLetterArray[rowVal+1, colVal+1]               
              
            maskArrayVal    =  (e0*mask[0] + e1*mask[1] + e2*mask[2] + 
                                e3*mask[3] + e4*mask[4] + e5*mask[5] + 
                                e6*mask[6] + e7*mask[7] + e8*mask[8] ) / 3.0                        
                         
            arrayCol = colVal - 1

            # rounding to the nearest integer so 0.33 = 0 and 0.66 = 1
            maskLetterArray[arrayRow,arrayCol] = np.round(maskArrayVal) 
                                                                                                                                        
            colVal = colVal + 1

        rowVal = rowVal + 1
        colVal = 1
                                        
    return maskLetterArray

####################################################################################################
####################################################################################################
#
# Procedure to convert the 2x2 array produced by maskLetter into a list and return the list 
#
####################################################################################################
####################################################################################################

def convertArrayToList(maskLetterArray, gridHeight, gridWidth):

    maskLetterList = list()

    for row in range(gridHeight):  #  Number of rows in a masked input grid
        for col in range(gridWidth):  # number of columns in a masked input grid
            localGridElement = maskLetterArray[row,col] 
            maskLetterList.append(localGridElement)   

    return (maskLetterList)
    

def applyMask (trainingDataList, inputArrayLength, mask, idx, len_gb1_classes, gridHeight, gridWidth):
    # Create a padded version of this letter
    # (Expand boundaries by one pixel all around)
    expandedLetterArray = list()
    expandedLetterArray = expandLetterBoundaries (trainingDataList, 11, 11, len_gb1_classes)# skip the 1st 9 elements
            
    # Optional print/debug
    #        printExpandedLetter (expandedLetterArray)
                
    maskLetterArray = maskLetterFunc(expandedLetterArray, mask[idx], gridHeight, gridWidth)
    maskLetterList = convertArrayToList(maskLetterArray,gridHeight, gridWidth) 
    return maskLetterList
                             
###############################################################################
#******************************************************************************
###############################################################################
#
# The MAIN module comprising of calls to:
#   (1) Welcome
#   (2) Obtain neural network size specifications for a three-layer network consisting of:
#       - Input layer
#       - Hidden layer
#       - Output layer (all the sizes are currently hard-coded to two nodes per layer right now)
#   (3) Initialize connection weight values
#       - w: Input-to-Hidden nodes
#       - v: Hidden-to-Output nodes
#   (4) Compute a feedforward pass in two steps
#       - Randomly select a single training data set
#       - Input-to-Hidden
#       - Hidden-to-Output
#       - Compute the error array
#       - Compute the new Summed Squared Error (SSE)
#   (5) Perform a single backpropagation training pass
#
###############################################################################
#******************************************************************************
###############################################################################


def main():

###############################################################################
# Obtain unit array size in terms of array_length (M) and layers (N)
###############################################################################

# This calls the procedure 'welcome,' which just prints out a welcoming message. 
# All procedures need an argument list. 
# This procedure has a list, but it is an empty list; welcome().

    directory = str("C:/EDUCATION/NWU/490 - Deep Learning/Week 7")
    print ("Results will be in: " + directory)
    
    os.chdir(directory)
    #random.seed(1) # set to ensure same set of random numbers across separate runs
    welcome()

    ###########################################################################
    # STEP 1 - Run the previously trained NN    
    ###########################################################################
    mode = input("[0]=Train GB1 only, [1]=Train Complex NN, [2]=Train Complex NN with Masking, [3]=Run trained GB1 and exit:")
    if mode > 0:
        gb1_output = runTrainedNetwork ()
        if mode == 3:
            sys.exit()

    ###########################################################################
    # Parameter default definitions, to be replaced with user inputs
    ###########################################################################
    alpha = 1.0
    eta = 1    
    epsilon = 0.01    
    
    ###########################################################################
    # Set iterations here
    ###########################################################################
    
    maxNumIterations = 200000    

    ###########################################################################
    # Override defaults by user inputs
    ###########################################################################
    
    #alpha = input ("Enter alpha value: ")
    #eta = input ("Enter eta value: ")
    noiseLevel = input("Enter noise level %: ")
    noiseLevel = noiseLevel/100.00
    #maxNumIterations = input ("Enter max iterations")
    #epsilon = input("Enter eplsilon")
    filename = "ABCD_26_VARS"

    # specify an input mask
    mask, gridHeight, gridWidth = masks()
       
    #Read the 81 element input training array for all 26 letters and variants
    # and associate classes - see digitizeCSVpattern(..)
    # ALL_26 option means 26 output classes - reads <filename>_classes_26.csv, otherwise GROUPS (default) means 
    # read the <filename>_classes.csv file to get groups
    if mode > 0:
        trainingData, numTrainingDataSets, outputNodes = readTrainingData(filename, "ALL_26")
        if mode == 2:
            trainingData, len_inputs = maskedTrainingData (trainingData, mask)
    else:        
        trainingData, numTrainingDataSets, outputNodes = readTrainingData(filename)
    
    # Define the variable arraySizeList, which is a list. It is initially an empty list. 
    # Its purpose is to store the size of the array.

    arraySizeList = list() # empty list

    # Obtain the actual sizes for each layer of the network
    len_inputs = len(trainingData[1]) - 2 #81

    len_gb1_classes = 0
    
    # set num input nodes to 90 (81+9) the 9 is for the number of classes
    if mode > 1: # the trained network has been run and its output array is available
        gb1_classes = gb1_output[:] # copy the output of the previously run trained NN
        len_gb1_classes = len(gb1_classes[1])
        # insert the 9 element class to the beginning of each training data row
        for i in range (len(trainingData)):
            trainingData[i]=np.insert(trainingData[i],0, gb1_classes[i])
            #for j in range (len_gb1_classes):
            #    trainingData[i].insert(j,gb1_classes[i][j])
        
        len_inputs = len(trainingData[1]) - 2 

    arraySizeList = obtainNeuralNetworkSizeSpecs (len_inputs, outputNodes)        
     
# Unpack the list; ascribe the various elements of the list to the sizes of different network layers    
    inputArrayLength = arraySizeList [0]
    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]

    SSE = np.zeros((outputArrayLength)) 
    LetterIters = np.zeros((outputArrayLength))

####################################################################################################
# Initialize the weight arrays for two sets of weights; w: input-to-hidden, and v: hidden-to-output
####################################################################################################                

#
# The wWeightArray is for Input-to-Hidden
# The vWeightArray is for Hidden-to-Output

    wWeightArraySizeList = (inputArrayLength, hiddenArrayLength)
    vWeightArraySizeList = (hiddenArrayLength, outputArrayLength)
    biasHiddenWeightArraySize = hiddenArrayLength
    biasOutputWeightArraySize = outputArrayLength        

# The node-to-node connection weights are stored in a 2-D array
    wWeightArray = initializeWeightArray (wWeightArraySizeList)
    vWeightArray = initializeWeightArray (vWeightArraySizeList)

# The bias weights are stored in a 1-D array         
    biasHiddenWeightArray = initializeBiasWeightArray (biasHiddenWeightArraySize)
    biasOutputWeightArray = initializeBiasWeightArray (biasOutputWeightArraySize) 

 

           
################################################################################
# Before we start training, get a baseline set of outputs, errors, and SSE 
################################################################################
           
    # Add the 9 element gray box (GB1) big shape class represented by a 9 
    # element array this will be obtained from the GB1 output which we read in 
    # from the previously trained GB1 NN - this now has 92 elements
    if mode > 0: #92 (9+81+2)
        if mode ==1:
            trainingDataList = [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0, ' ']
        if mode == 2:
            trainingDataList = [0]*(inputArrayLength+2)
            trainingDataList[len(trainingDataList)-2] = 0
            trainingDataList[len(trainingDataList)-1] = ' '
    else: #83 (81+2)
        trainingDataList = [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0, ' ']


    print ' '
    print '  Before training:'
    
    ComputeOutputsAcrossAllTrainingData (alpha, arraySizeList, numTrainingDataSets, wWeightArray, 
biasHiddenWeightArray, vWeightArray, biasOutputWeightArray, trainingData, mode)                           
                                             
          
################################################################################
# Next step - Obtain a single set of randomly-selected training values for 
# alpha-classification 
################################################################################
  
    iteration = 0
    
    while iteration < maxNumIterations:           

# Increment the iteration count
        iteration = iteration +1
    
# For any given pass, we re-initialize the training list
        # Add the 9 element gray box (GB1) big shape class represented by a 
        # 9 element array this will be obtained from the GB1 output which we 
        # read in from the previously trained GB1 NN - this now has 92 elements
        #if mode > 0: #92 (9+81+2)
        #    trainingDataList = [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0, ' ']
        #    # insert 0's for the GB1 output at the beginning of the array
        #    for i in range(len_gb1_classes):
        #        trainingDataList.insert(0,0)        
        #else: #83 (81+2)
        #    trainingDataList = [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0, ' ']

        inputDataList = []                                
                                                                                          
        # Randomly select one of the training sets; the inputs will be randomly 
        # assigned to 0 or 1
        trainingDataList = obtainRandomAlphabetTrainingValues (trainingData, numTrainingDataSets) 

        trainingDataRow = trainingDataList[:]
        for node in range(inputArrayLength): 
            trainingDataCell = float(trainingDataRow[node])
            
            ####################################################################
            # Inject Noise here (only the 9x9 letter grid is considered leaving 
            # out the GB1 outut at the  beginning of the array
            ####################################################################
            
            if noiseLevel > 0:
                randNum =  random.random()
                if randNum < noiseLevel: # Introduce noise
                    if mode > 0: # if complex NN
                        if node > 8: # leave the class inputs alone
                            if trainingDataCell == 0: 
                                trainingDataCell = 1
                            else: 
                                trainingDataCell = 0

                    else:
                        if trainingDataCell == 0: 
                            trainingDataCell = 1
                        else: 
                            trainingDataCell = 0 
                    
            inputDataList.append(float(trainingDataCell))
        
                        
        desiredOutputArray = np.zeros(outputArrayLength)    # iniitalize the output array with 0's
        classIdx = len(trainingDataList) - 2                # classId should evaluate to 81 for 9x9 grid
        desiredClass = int(trainingDataList[classIdx])           # identify the desired class
        desiredOutputArray[desiredClass] = 1                # set the desired output for that class to 1

                  
        ########################################################################
        # Compute a single feed-forward pass and obtain the Actual Outputs
        ########################################################################
                

        hiddenArray = ComputeSingleFeedforwardPassFirstStep (alpha, arraySizeList, inputDataList, 
        wWeightArray, biasHiddenWeightArray)
    
#        print ' '
#        print ' The hidden node activations are:'
#        print hiddenArray

        outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, arraySizeList, hiddenArray,
        vWeightArray, biasOutputWeightArray)
    
#        print ' '
#        print ' The output node activations are:'
 #       print outputArray    

#  Optional alternative code for later use:
#  Assign the hidden and output values to specific different variables
#    for node in range(hiddenArrayLength):    
#        actualHiddenOutput[node] = actualAllNodesOutputList [node]
    
#    for node in range(outputArrayLength):    
#        actualOutput[node] = actualAllNodesOutputList [hiddenArrayLength + node]
 
# Initialize the error array
        errorArray = np.zeros(outputArrayLength) 
    
# Determine the error between actual and desired outputs        
        newSSE = 0.0
        for node in range(outputArrayLength):  #  Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node]*errorArray[node]        

#        print ' '
#        print ' The error values are:'
#        print errorArray   
        
# Print the Summed Squared Error  
#        print 'Initial SSE = %.6f' % newSSE
#        SSE = newSSE

         
          
####################################################################################################
# Perform backpropagation
####################################################################################################                
                

# Perform first part of the backpropagation of weight changes    
        newVWeightArray = BackpropagateOutputToHidden (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray, vWeightArray)
        newBiasOutputWeightArray = BackpropagateBiasOutputWeights (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray) 

# Perform first part of the backpropagation of weight changes       
        newWWeightArray = BackpropagateHiddenToInput (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
        inputDataList, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray)

        newBiasHiddenWeightArray = BackpropagateBiasHiddenWeights (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
        inputDataList, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray)  
    
                    
# Assign new values to the weight matrices
# Assign the old hidden-to-output weight array to be the same as what was returned from the BP weight update
        vWeightArray = newVWeightArray[:]
    
        biasOutputWeightArray = newBiasOutputWeightArray[:]
    
# Assign the old input-to-hidden weight array to be the same as what was returned from the BP weight update
        wWeightArray = newWWeightArray[:]  
    
        biasHiddenWeightArray = newBiasHiddenWeightArray[:] 
    
# Compute a forward pass, test the new SSE                                                                                
                                                                                                                                    
        hiddenArray = ComputeSingleFeedforwardPassFirstStep (alpha, arraySizeList, inputDataList, 
        wWeightArray, biasHiddenWeightArray)
    
#    print ' '
#    print ' The hidden node activations are:'
#    print hiddenArray

        outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, arraySizeList, hiddenArray,
        vWeightArray, biasOutputWeightArray)
    
#    print ' '
#    print ' The output node activations are:'
#    print outputArray    

    
# Determine the error between actual and desired outputs

        newSSE = 0.0
        for node in range(outputArrayLength):  #  Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node]*errorArray[node]        

#        print ' '
#        print ' The error values are:'
#        print errorArray   
        
# Print the Summed Squared Error  
#        print 'Previous SSE = %.6f' % SSE
#        print 'New SSE = %.6f' % newSSE 
    
#        print ' '
#        print 'Iteration number ', iteration
        #iteration = iteration + 1

        #save the parameters to the output csv file
        #save_to_outfile (wr, trainingDataList[letterIdx], newSSE, hiddenArray, outputArray)
        if newSSE < epsilon:          
            SSE[int(trainingDataList[classIdx])]=1
            sum_sse = sum(SSE)            
            if (sum_sse >= outputArrayLength):
                break
        else:
            idx = int(trainingDataList[classIdx])
            LetterIters[idx] += 1
    print 'Out of while loop at iteration ', iteration 
    
####################################################################################################
# After training, get a new comparative set of outputs, errors, and SSE 
####################################################################################################                           

    print ' '
    print '  After training:'                  
                                                   
    # create an output csv file to record the execution
    outfile, wr = create_outfile(filename,arraySizeList[1], alpha,eta,epsilon,maxNumIterations,noiseLevel, cnn_mode=mode)
    
    ComputeOutputsAcrossAllTrainingData (alpha, arraySizeList, numTrainingDataSets, wWeightArray, 
biasHiddenWeightArray, vWeightArray, biasOutputWeightArray, trainingData, training=0, writer=wr, cnn_mode=mode) 

    close_outfile (outfile,wr, hiddenArrayLength, iteration, alpha, eta, epsilon,LetterIters)
    
    if mode == 0:
        #
        # Use pickle to write to output files and store parameters
        #
        output = open("arraySizeList.pkl.txt", 'wb')
        pk.dump(arraySizeList, output)
        output.close()
    
        output = open("wWeightArray.pkl.txt", 'wb')
        pk.dump(wWeightArray,output)
        output.close()
        
        output = open("biasHiddenWeightArray.pkl.txt", 'wb')
        pk.dump(biasHiddenWeightArray,output, -1)
        output.close()
        
        output = open("vWeightArray.pkl.txt", 'wb')
        pk.dump(vWeightArray,output)
        output.close()
        
        output = open("biasOutputWeightArray.pkl.txt", 'wb')
        pk.dump(biasOutputWeightArray,output)
        output.close()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
                                              
####################################################################################################
# Conclude specification of the MAIN procedure
####################################################################################################                
    
if __name__ == "__main__": main()

####################################################################################################
# End program
#################################################################################################### 

