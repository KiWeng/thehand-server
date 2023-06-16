import keras.utils
import tensorflow as tf
from mne import filter

from fake import EegoDriver
from utils import record

freqs = (50.0, 100.0, 150.0, 200.0, 250, 300, 350, 400, 450, 500, 550, 600)
l_freq, h_freq = 8, 500
sample_frequency = 2000
batch_size = 32


# TODO: test this
def calibrate(board, duration, channels, new_model_path,
              model_path='../assets/saved_model/finetuned'):
    bipolar_data = record(board, duration, channels)[:, :12]  # TODO

    stds = bipolar_data.std(axis=0)
    filtered = filter.notch_filter((bipolar_data[:, :12] / stds).T, sample_frequency, freqs,
                                   filter_length="1000ms", trans_bandwidth=8.0,
                                   picks=range(12), verbose=False)
    filt = filter.create_filter(filtered, sample_frequency, l_freq, h_freq, verbose=False)
    filtered = filter._overlap_add_filter(filtered, filt, picks=[i for i in range(12)]).T

    """
    the calibration session will be:
        for each finger and a fist (total of 6 gestures):
            2s of idle, 3s of gesture (0.5 + 2 + 0.5)
                     /--------\
            --------/          \
        when calibrating the model, 2s of idle and 2s of gesture will be used
    """
    dataset_all = None
    for i in range(6):
        input_dataset = keras.utils.timeseries_dataset_from_array(filtered, None,
                                                                  sequence_length=800,
                                                                  sequence_stride=10, batch_size=1)
        idle_input_ds = input_dataset.take(400)
        active_input_ds = input_dataset.skip(500).take(400)
        gesture = [0, 0, 0, 0, 0]
        if i == 0:
            gesture = [16, 0, 0, 0, 0]
        elif i == 1:
            gesture = [0, 16, 0, 0, 0]
        elif i == 2:
            gesture = [0, 0, 16, 0, 0]
        elif i == 3:
            gesture = [0, 0, 0, 16, 0]
        elif i == 4:
            gesture = [0, 0, 0, 0, 16]
        elif i == 5:
            gesture = [16, 16, 16, 16, 16]

        active_output_ds = keras.utils.timeseries_dataset_from_array(
            [gesture for i in range(400)])
        idle_output_ds = keras.utils.timeseries_dataset_from_array(
            [[0, 0, 0, 0, 0] for i in range(400)])

        idle_ds = tf.data.Dataset.zip((idle_input_ds, idle_output_ds))
        active_ds = tf.data.Dataset.zip((active_input_ds, active_output_ds))
        if dataset_all is None:
            dataset_all = idle_ds
        else:
            dataset_all.append(idle_ds)
        dataset_all.append(active_ds)

        dataset_all.unbatch().batch(batch_size)

        model = tf.keras.models.load_model('../assets/saved_model/base')
        model.fit(dataset_all, epochs=5)
        model.save(new_model_path)

        print(type(bipolar_data))
        print(filtered.shape)

    return stds


if __name__ == "__main__":
    board = EegoDriver(2000)
    stds = calibrate(board, 30, "../assets/saved_model/new_calibrated")
    print(stds)
