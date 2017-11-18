from keras.utils import np_utils
import numpy as np
import pandas as pd

def getGroundTruthCSV(path, n):
    data_ground_truth = pd.read_csv(path)
    data_ground_truth = data_ground_truth.iloc[0:n, 1]
    y_data_ground_truth = data_ground_truth
    y_data_ground_truth = np.array(y_data_ground_truth)
    y_data_ground_truth = np_utils.to_categorical(data_ground_truth)

    return y_data_ground_truth