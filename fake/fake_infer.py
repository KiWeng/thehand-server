import numpy as np
from mne import filter
from scipy import signal

from model import EMGModel
from .fake_board import FakeBoard

stds = [
    0.000034,
    0.000019,
    0.000013,
    0.000030,  # TODO
    0.000012,
    0.000014,
    0.000031,
    0.000028,
    0.000036,
    0.000028,
    0.000051,
    0.000047,
]  # std value exclude extreme is 0.00003
notch_b, notch_a = signal.iirnotch(50, 30, 2000)
band_b, band_a = signal.butter(5, [8, 500], 'bandpass', fs=2000)


class FakeInfer:
    def __init__(self):
        self.board = FakeBoard()
        self.model = EMGModel()

    def get_infer(self):
        while True:
            if self.board.get_board_data_count() < 8 * 800:
                continue
            else:
                fake_data = self.board.get_current_board_data(8 * 800)[:12, ::8]
                break

        fake_data = fake_data[:, -800:]

        # filtered_data = signal.filtfilt(notch_b, notch_a, fake_data.T[:, :12], axis=0)
        # filtered_data = signal.filtfilt(band_b, band_a, filtered_data, axis=0)
        # normalized_data = filtered_data / stds

        sample_frequency = 2000
        freqs = (50.0, 100.0, 150.0, 200.0, 250, 300, 350, 400, 450, 500, 550, 600)
        l_freq, h_freq = 8, 500

        filtered = filter.notch_filter((fake_data.T[:800, :12] / stds).T, sample_frequency, freqs,
                                       picks=range(12))
        filt = filter.create_filter(filtered, sample_frequency, l_freq, h_freq)
        filtered = filter._overlap_add_filter(filtered, filt, picks=[i for i in range(12)])

        fake_data = np.expand_dims(filtered.T, axis=0)
        return self.model.predict(fake_data)

# if __name__ == "__main__":
# while True:
#     print(fake_data.shape)
#     print(model.predict(fake_data))
