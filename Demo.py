#A demo to illustrate how to train the PHNN and apply the PHNN to make a forecast



#Add the user's file path to the system default paths
import sys
sys.path.append('/Users/min/Desktop/git_min')


#Note:Old version tensorflow for experiments in the study
#from PHNN1 import PHNN_Model 
#Note:New version of tensorflow 
from PHNN2 import PHNN_Model 
from DataHandler import DataHandler

import numpy as np
import warnings
warnings.filterwarnings('ignore')



##Train the model##

#Set the path of the data
datapath=r'/Users/min/Desktop/git_min/data/NG_NEW.xlsx'

# Import the data and proccess the data  
data_handler = DataHandler(datapath)
phnn_model = PHNN_Model(data_handler)
trainX_1, valX_1, testX_1, trainX_2, valX_2, testX_2, trainY_1, valY_1, testY_1 = data_handler.prepare_datasets()

# Build the architecture of PHNN
phnn_model.define_model((4,4))
# Train the PHNN on training data
best_model = phnn_model.train_model(trainX_1, trainX_2, trainY_1, valX_1, valX_2, valY_1)




##Performance assessment##

#Conduct the prediction by the trained PHNN model 
prediction = best_model.predict([testX_1, testX_2])

#Recover the price series;
prediction_copies_array = np.repeat(prediction, 4, axis=-1)
pred = phnn_model.scaler.inverse_transform(np.reshape(
    prediction_copies_array, (len(prediction), 4)))[:, 3]
original_copies_array = np.repeat(testY_1, 4, axis=-1)
original = phnn_model.scaler.inverse_transform(
    np.reshape(original_copies_array, (len(testY_1), 4)))[:, 3]

#Import the metrics for assessment 
from evaluation import evaluation
#Assess the performance of PHNN
evaluation(original, pred)

