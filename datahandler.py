from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_and_preprocess_data()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_data, self.val_data, self.test_data, self.y_1, self.y_2, self.y_3 =self.split_data()

    def load_and_preprocess_data(self):
        # Load data
        data = pd.read_excel(self.file_path)

        # Select required columns
        data = data.iloc[2455:8260, [4, 3, 6, 5]]

        # Reset index
        data = data.reset_index(drop=True)

        return data

    def split_data(self):
        # Calculate split points
        test_split = round(len(self.data) * 0.2)
        train_split = round(len(self.data) * 0.3)
        val_split = train_split - test_split

        # Split data
        data_1 = self.data[:-train_split]
        data_3 = self.data[-test_split-16:]
        data_2 = self.data[-train_split-16:-test_split]

        # Normalize data
        data_1 = self.scaler.fit_transform(np.array(data_1))
        data_2 = self.scaler.transform(np.array(data_2))
        data_3 = self.scaler.transform(np.array(data_3))

        y_1 = data_1[16:, 2]
        y_2 = data_2[16:, 2]
        y_3 = data_3[16:, 2]

        return data_1, data_2, data_3, y_1, y_2, y_3

    def get_scaler(self):
        return self.scaler

    def create_long_term(self, data, window_size=16):
        r, c = data.shape
        x_data = None

        for k in range(r - window_size):
            cpat = data[np.array([k, k+5, k+10, k+15]), :]
            if x_data is None:
                x_data = cpat
            else:
                x_data = np.append(x_data, cpat, axis=0)

        new = x_data
        x = new.reshape(-1, 4, 4)

        return x

    def create_short_term(self, data, window_size=16):
        r, c = data.shape
        x_data = None

        for k in range(r - window_size):
            cpat = data[np.array([k+12, k+13, k+14, k+15]), :]
            if x_data is None:
                x_data = cpat
            else:
                x_data = np.append(x_data, cpat, axis=0)

        new = x_data
        x = new.reshape(-1, 4, 4)

        return x

    def prepare_datasets(self):
        # Create datasets
        trainX_1 = self.create_short_term(self.train_data)
        valX_1 = self.create_short_term(self.val_data)
        testX_1 = self.create_short_term(self.test_data)

        trainX_2 = self.create_long_term(self.train_data)
        valX_2 = self.create_long_term(self.val_data)
        testX_2 = self.create_long_term(self.test_data)

        # Shuffle the indices for training data
        index = [i for i in range(len(trainX_1))]
        np.random.seed(1)
        np.random.shuffle(index)

        # Apply the shuffled indices to the training data
        trainX_1 = trainX_1[index]
        trainX_2 = trainX_2[index]
        trainY_1 = self.y_1[index]
        valY_1 = self.y_2
        testY_1 = self.y_3

        return trainX_1, valX_1, testX_1, trainX_2, valX_2, testX_2, trainY_1, valY_1, testY_1
