import time

import numpy as np


class EegoDriver:

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.data = np.genfromtxt(
            r"C:\Users\kwzh\PythonProjects\thehand-server\tmp\bipolar_recording_1684499974.8314939.csv",
            delimiter=" ")
        self.init_time_ns = time.time_ns()

    def get_data_of_size(self, size):
        current_time_ns = time.time_ns()
        return_pos = int((current_time_ns - self.init_time_ns) / 1e9 * 2000)
        return_len = min(size, return_pos)
        return self.data[return_pos - return_len:return_pos], \
            self.data[return_pos - return_len:return_pos]
