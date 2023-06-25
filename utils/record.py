import time

import numpy as np


def record(board, duration, channels=[i for i in range(12)]):
    """
    :param board:
    :param duration: int: seconds
    :param channels:
    :return: [duration * sampling_rate, channels]
    """
    bipolar_data_all = None
    board.get_data()  # drop previous data
    for i in range(duration):
        time.sleep(1)
        data = board.get_data()
        print(data[1].shape)
        if data is None:
            time.sleep(1)
        elif bipolar_data_all is None:
            bipolar_data = data[1]
            bipolar_data_all = bipolar_data
        else:
            bipolar_data = data[1]
            bipolar_data_all = np.concatenate((bipolar_data_all, bipolar_data), axis=0)

    return bipolar_data_all
