import time

import numpy as np


class EegoDriver:

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.data = np.genfromtxt(
            # r"..\tmp\calibration_record_1689067937.4199173.csv",
            # r"..\tmp\calibration_record_1691115431.870155.csv",
            r"..\tmp\calibration_record_1691114520.2038014.csv",
            # r"..\tmp\calibration_record_1691114988.0299563.csv",  # 12
            # r"..\tmp\calibration_record_1691136366.124025.csv",  # 15-4
            # r"..\tmp\calibration_record_1691136880.0698347.csv",  # 15-2
            # r"..\tmp\calibration_record_1691135900.9566607.csv",  # 15-1
            # r"..\tmp\calibration_record_1691135368.3412945.csv",  # 15-0
            delimiter=",")
        self.init_time_ns = time.time_ns()
        self.max_len = self.data.shape[0]
        self.last_pos = 0

    def get_data_of_size(self, size):
        current_time_ns = time.time_ns()
        return_pos = int((current_time_ns - self.init_time_ns) / 1e9 * 2000)

        return_pos = (return_pos - size) % self.max_len
        return_len = min(size, return_pos)

        print(return_pos, return_len, size, self.max_len)

        return self.data[return_pos - return_len:return_pos], \
            self.data[return_pos - return_len:return_pos]

    def get_data(self):
        current_time_ns = time.time_ns()
        return_pos = int((current_time_ns - self.init_time_ns) / 1e9 * 2000)
        last_pos = self.last_pos
        self.last_pos = return_pos
        return self.data[last_pos:return_pos], \
            self.data[last_pos:return_pos]


if __name__ == "__main__":
    fb = EegoDriver(2000)

    for i in range(100):
        time.sleep(1)
        data = fb.get_data()
        print(data[1].shape)
