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

    def get_current_board_data(self, num_samples: int) -> ndarray:
        """Get specified amount of assets or less if there is not enough assets

        :param num_samples: max number of samples
        :type num_samples: int
        :return: latest assets from a board
        :rtype: ndarray
        """
        if self.check_time:
            current_time_ns = time.time_ns()
            calculated_sample_len = int(
                (current_time_ns - self.init_time_ns) / 1e9 * self.sampling_rate)
            print(current_time_ns - self.init_time_ns)
            print(calculated_sample_len)
            num_samples = num_samples if num_samples <= calculated_sample_len else calculated_sample_len
        else:
            self.check_time = False

        return np.random.rand(self.channel_num, num_samples)

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
        self.buffer_cleaned_time = current_time_ns

        return np.random.rand(self.channel_num, min(calculated_sample_len, num_samples))

    def get_board_data_count(self):
        current_time_ns = time.time_ns()
        calculated_sample_len = int(
            (current_time_ns - self.buffer_cleaned_time) / 1e9 * self.sampling_rate)
        return calculated_sample_len


if __name__ == "__main__":
    board = FakeBoard()
    for i in range(16):
        fake_data = board.get_board_data(8 * 16000)
        print(fake_data.shape)
        time.sleep(1)
