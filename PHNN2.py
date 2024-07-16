#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, median_absolute_error
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow import optimizers
from tensorflow.keras import Sequential, callbacks
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.optimizers import Adam
import keras
from tensorflow.keras.layers import LSTM
from keras.models import Model
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


class PHNN_Model:
    def __init__(self, data_handler):
        self.model = None
        self.data_handler = data_handler
        self.scaler = data_handler.get_scaler()

    def define_model(self, input_shape):
        # Define the first model
        modelx = Sequential()
        modelx.add(LSTM(128, input_shape=input_shape, return_sequences=True))
        modelx.add(Dropout(0.2))
        modelx.add(LSTM(128, return_sequences=False))
        modelx.add(Dropout(0.2))
        modelx.add(Dense(16))
        modelx.add(Flatten())

        # Define the second model
        modely = Sequential()
        modely.add(LSTM(128, input_shape=input_shape, return_sequences=True))
        modely.add(Dropout(0.2))
        modely.add(LSTM(128, return_sequences=False))
        modely.add(Dropout(0.2))
        modely.add(Dense(16))
        modely.add(Flatten())

        # Merge the outputs of the two models
        dummy_input_x = Input(shape=(4, 4))
        dummy_input_y = Input(shape=(4, 4))

        dummy_output_x = modelx(dummy_input_x)
        dummy_output_y = modely(dummy_input_y)
        
        merged = Concatenate()([dummy_output_x, dummy_output_y])
        z = Dense(128)(merged)
        z = Dropout(0.2)(z)
        z = Dense(16)(z)
        z = Dense(1)(z)

        # Create the final model
        self.model = Model(inputs=[dummy_input_x, dummy_input_y], outputs=z)
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

    def train_model(self, train_data1, train_data2, train_labels, val_data1, val_data2, val_labels, batch_sizes=[16, 25, 32], epochs=300):
        best_val_loss = float('inf')
        best_model = None
        best_batch_size = None

        # Iterate over different batch sizes
        for batch_size in batch_sizes:
            # Define callback functions
            es = EarlyStopping(monitor='val_loss', min_delta=0,
                               patience=80, mode='min', verbose=1)
            bm = ModelCheckpoint(
                f'best_model_batch_{batch_size}.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

            # Train the model with the current batch size
            history = self.model.fit([train_data1, train_data2], train_labels, batch_size=batch_size, epochs=epochs,
                                     verbose=1, validation_data=([val_data1, val_data2], val_labels),
                                     callbacks=[es, bm])

            # Load the best model for this batch size
            self.model.load_weights(f'best_model_batch_{batch_size}.keras')

            # Evaluate the model on the validation set
            val_loss = self.model.evaluate(
                [val_data1, val_data2], val_labels, verbose=0)

            # Check if this model has the best validation loss so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                best_batch_size = batch_size

            # Print the validation loss for this batch size
            print(f"Batch size {batch_size}: Validation loss = {val_loss}")

        # Save the best model
        if best_model is not None:
            best_model.save(f"best_model_batch_{best_batch_size}.keras")
            print(f"Best model with batch size {best_batch_size} saved.")
        else:
            print("No model was found to be better than the initial model.")

        return best_model
