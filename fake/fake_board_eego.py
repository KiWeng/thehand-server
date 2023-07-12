import time

import numpy as np


class EegoDriver:

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.data = np.genfromtxt(
            # r"C:\Users\kwzh\PythonProjects\thehand-server\tmp\bipolar_recording_1684500719.653217.csv",
            r"C:\Users\kwzh\PythonProjects\thehand-server\calibration_record_1689067942.9628341_filtered.csv",
            delimiter=" ")
        self.init_time_ns = time.time_ns()
        self.last_pos = 0

    def get_data_of_size(self, size):
        current_time_ns = time.time_ns()
        return_pos = int((current_time_ns - self.init_time_ns) / 1e9 * 2000)
        return_len = min(size, return_pos)
        return self.data[return_pos - return_len:return_pos], \
            self.data[return_pos - return_len:return_pos]

    def get_data(self):
        current_time_ns = time.time_ns()
        return_pos = int((current_time_ns - self.init_time_ns) / 1e9 * 2000)
        last_pos = self.last_pos
        # print((last_pos, return_pos))
        self.last_pos = return_pos
        return self.data[last_pos:return_pos], \
            self.data[last_pos:return_pos]


if __name__ == "__main__":
    fb = EegoDriver(2000)

    for i in range(100):
        time.sleep(1)
        data = fb.get_data()
        print(data[1].shape)
