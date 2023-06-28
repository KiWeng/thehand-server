import unittest

from fake import EegoDriver
from model import EMGModel
from utils import filter_data, make_calibration_ds
from utils.record import record


class TestCalibration(unittest.TestCase):

    def test_record(self):
        duration = 3
        board = EegoDriver(sampling_rate=2000)
        bipolar_data, last_record_time = record(board, duration)
        bipolar_data = bipolar_data[:, :12]
        print(bipolar_data.shape)
        self.assertEqual(bipolar_data.shape[-1], 12)

    def test_make_ds(self):
        duration = 30
        board = EegoDriver(sampling_rate=2000)
        bipolar_data, last_record_time = record(board, duration)
        bipolar_data = bipolar_data[:, :12]
        current_stds = bipolar_data.std(axis=0)
        filtered_data = filter_data((bipolar_data / current_stds).T).T
        ds = make_calibration_ds(filtered_data)
        assert bipolar_data.shape == filtered_data.shape

    def test_calibration(self):
        model = EMGModel()
        duration = 30
        board = EegoDriver(sampling_rate=2000)
        bipolar_data, last_record_time = record(board, duration)
        bipolar_data = bipolar_data[:, :12]
        current_stds = bipolar_data.std(axis=0)
        filtered_data = filter_data((bipolar_data / current_stds).T).T
        ds = make_calibration_ds(filtered_data)
        model.calibrate(ds, "../assets/saved_model/tmp")
