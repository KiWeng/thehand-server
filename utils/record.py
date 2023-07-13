import time

import numpy as np


def record(board, duration, channels=None):
    """
    :param board:
    :param duration: int: seconds
    :param channels:
    :return: [duration * sampling_rate, channels]
    """
    if channels is None:
        channels = [i for i in range(12)]
    last_record_time = 0
    bipolar_data_all = None
    board.get_data()  # drop previous data
    for i in range(duration):
        time.sleep(1)
        data = board.get_data()
        last_record_time = time.time()

        # print(f'record i {i}')

        if data is None:
            time.sleep(1)
        elif bipolar_data_all is None:
            bipolar_data = data[1]
            bipolar_data_all = bipolar_data
        else:
            bipolar_data = data[1]
            bipolar_data_all = np.concatenate((bipolar_data_all, bipolar_data), axis=0)

    new_data = np.genfromtxt(r'..\tmp\calibration_record_1689067937.4199173.csv', delimiter=',')
    new_data = np.pad(new_data, ((233, 0), (0, 0)), 'mean')
    new_data = new_data[:70000, :12]
    bipolar_data_all = new_data
    assert new_data.shape == (70000, 12)

    return bipolar_data_all, last_record_time
