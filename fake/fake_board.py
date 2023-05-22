import time

import numpy as np
from numpy import ndarray


class FakeBoard:
    """Fake a board
    :param channel_num:
    :type channel_num: int
    :param sampling_rate:
    :type sampling_rate: int

    """

    def __init__(self, channel_num: int = 92, sampling_rate: int = 16000) -> None:
        # sampling rate for ant neuro board is 16000 and 92 channels
        self.channel_num = channel_num
        self.sampling_rate = sampling_rate
        self.check_time = True
        self.init_time_ns = time.time_ns()
        self.buffer_cleaned_time = self.init_time_ns
        # data = np.genfromtxt("../assets/example_data/李存波_2022-11-12_emg_noabs.csv",
        #                      delimiter=",")
        self.data = np.genfromtxt(
            r"C:\Users\kwzh\PythonProjects\thehand-server\tmp\bipolar_recording_1684499974.8314939.csv",
            delimiter=" ")[:, :12]

    def get_current_board_data(self, num_samples: int) -> ndarray:
        """Get specified amount of assets or less if there is not enough assets

        :param num_samples: max number of samples
        :type num_samples: int
        :return: latest assets from a board
        :rtype: ndarray: channels * samples
        """
        current_time_ns = time.time_ns()
        if self.check_time:
            calculated_sample_len = int(
                (current_time_ns - self.init_time_ns) / 1e9 * self.sampling_rate)
            print(current_time_ns - self.init_time_ns)
            print(calculated_sample_len)
            num_samples = num_samples if num_samples <= calculated_sample_len else calculated_sample_len
        else:
            self.check_time = False

        start_sample = int((current_time_ns - self.init_time_ns) / (1e9 / 2000))
        data = self.data[start_sample:start_sample + int(num_samples / 8)]

        data_expanded = np.array([e for e in data for _ in range(8)])
        data_expanded = data_expanded.T
        data_expanded = np.pad(data_expanded, ((0, 92 - 12), (0, 0)), 'constant', constant_values=0)

        assert np.array_equal(data, data_expanded[:12, ::8].T)
        self.buffer_cleaned_time = current_time_ns

        return data_expanded

        # return data

    def get_board_data(self, num_samples: int) -> ndarray:
        # TODO
        """Get board data and remove data from ringbuffer

        :param num_samples: max number of samples
        :type num_samples: int
        :return: latest assets from a board
        :rtype: ndarray
        """

        current_time_ns = time.time_ns()
        calculated_sample_len = int(
            (current_time_ns - self.buffer_cleaned_time) / 1e9 * self.sampling_rate)

        data = self.data[int((self.buffer_cleaned_time - self.init_time_ns) / (1e9 / 2000)):
                         int((current_time_ns - self.init_time_ns) / (1e9 / 2000))]
        print(data.shape)

        data_expanded = np.array([e for e in data for _ in range(8)])
        data_expanded = data_expanded.T
        data_expanded = np.pad(data_expanded, ((0, 92 - 12), (0, 0)), 'constant', constant_values=0)

        assert np.array_equal(data, data_expanded[:12, ::8].T)
        self.buffer_cleaned_time = current_time_ns

        return data_expanded

    def get_board_data_count(self):
        current_time_ns = time.time_ns()
        calculated_sample_len = int(
            (current_time_ns - self.buffer_cleaned_time) / 1e9 * self.sampling_rate)
        return calculated_sample_len


if __name__ == "__main__":
    board = FakeBoard()
    for i in range(16):
        # fake_data = board.get_board_data(8 * 16000)
        fake_data = board.get_current_board_data(8 * 800)
        print(fake_data.shape)
        time.sleep(1)
