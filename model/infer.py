import time

# from fake import EegoDriver
import numpy as np
import tensorflow as tf
from mne import filter

from utils import KalmanFilter

stds = [
    0.000034,
    0.000019,
    0.000013,
    0.300000,
    0.000012,
    0.000014,
    0.000031,
    0.000028,
    0.000036,
    0.000028,
    0.000051,
    0.000047,
]  # std value exclude extreme is 0.00003

sample_frequency = 2000
freqs = (50.0, 100.0, 150.0, 200.0, 250, 300, 350, 400, 450, 500, 550, 600)
l_freq, h_freq = 8, 500

# 定义观测矩阵和状态转移矩阵
H = np.array([[1]])
F = np.array([[1]])

# 定义初始状态向量和协方差矩阵
x = np.array([0])
P = np.eye(1)

# 定义噪声协方差矩阵
Q = np.array([[1]])
R = np.array([[1]])

# end of kalman

sample_len = 2000


class EMGModel:
    def __init__(self, board, model_path='../assets/saved_model/finetuned'):
        self.model = tf.keras.models.load_model(model_path)
        self.board = board
        self.kfs = [KalmanFilter(F=F, H=H, Q=Q, R=R, P=P, x=x) for i in range(5)]

    def predict(self, data):
        return self.model.predict(data)

    def infer(self):
        def step(driver):
            data = driver.get_data_of_size(sample_len)
            bipolar_data = data[1]
            while bipolar_data.shape[0] < sample_len:
                time.sleep(0.2)
                data = driver.get_data_of_size(sample_len)
                bipolar_data = data[1]

            filtered = filter.notch_filter((bipolar_data[:, :12] / stds).T, sample_frequency, freqs,
                                           filter_length="1000ms", trans_bandwidth=8.0,
                                           picks=range(12), verbose=False)
            filt = filter.create_filter(filtered, sample_frequency, l_freq, h_freq, verbose=False)
            filtered = filter._overlap_add_filter(filtered, filt, picks=[i for i in range(12)]).T

            # print(filtered.shape)
            delay = 100
            # print(filtered[:-delay][-800:, :].std(axis=0))

            # print(normalized_data[-1:, :])

            normalized_data = np.expand_dims(filtered[:-delay][-800:, :], axis=0)
            '''
            TODO: There are some problems:
            1. Some times some channels are sinusoidal so it will create large spikes
             at the start&end of the signal, so head and tail must be cut
            2. A overlong segment is needed to have a good filtered result
            '''

            '''
            20230615
            The problem still remains, by changing the trans_bandwidth & filter_length the lag have been
            greatly reduced but still remains (delay can be set to 0 but will the quality of the filtered
            data will be worsen)
            Things that affect filtered data quality:
                change the filtered_slice length (longer is better)
                change the filter_length&trans_bandwidth in notch_filter parameters
                padding
            '''

            prediction = self.model.predict(normalized_data)  # (samples, channels)

            filtered_pred = []
            for i in range(len(prediction[0])):
                self.kfs[i].predict()
                self.kfs[i].update(prediction[0][i])
                filtered_pred.append(self.kfs[i].x[0])

            print([filtered_pred])

            return np.asarray([filtered_pred])

        # too slow in test
        # if board.get_board_data_count() < 8 * 800:
        #     continue

        # elegant, but slower, fuck
        # prediction = await asyncio.to_thread(infer_step, board)
        prediction = step(self.board)
        return prediction


if __name__ == "__main__":
    new_model = EMGModel()
    new_model.model.summary()
