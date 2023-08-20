import warnings
from typing import Optional

import keras.utils
import tensorflow as tf
from mne import filter


def filter_data(data, sample_frequency=2000, l_freq=8, h_freq=500,
                freqs=(50.0, 100.0, 150.0, 200.0, 250, 300, 350, 400, 450, 500, 550, 600)):
    '''
    :param data:  [channels, num_samples]
    :param sample_frequency:
    :param l_freq:
    :param h_freq:
    :param freqs:
    :return: [channels, num_samples]
    '''
    # supress DeprecationWarning caused by mne filtering function
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)

    filtered = filter.notch_filter(
        data, sample_frequency, freqs, filter_length="1000ms",
        trans_bandwidth=8.0, picks=range(12), verbose=False)
    filt = filter.create_filter(filtered, sample_frequency, l_freq, h_freq, verbose=False)
    filtered = filter._overlap_add_filter(filtered, filt, picks=[i for i in range(12)])
    return filtered


# TODO: test this
def make_calibration_ds(filtered_data, gestures, batch_size=64):
    dataset_all: Optional[tf.data.Dataset] = None

    input_dataset = keras.utils.timeseries_dataset_from_array(
        filtered_data, None, sequence_length=800, sequence_stride=20, batch_size=1)

    """
    the calibration session will be:
        for each finger and a fist (total of 6 gestures):
            2s of idle, 3s of gesture (0.5 + 2 + 0.5)
                      --------
                     /        \
            --------/          \
        when calibrating the model, 2s of idle and 2s of gesture will be used
    """

    for i in range(len(gestures)):
        # TODO
        idle_input_ds = input_dataset.skip(i * 500 + 100).take(100)
        active_input_ds = input_dataset.skip(i * 500 + 350).take(100)

        gesture = gestures[i] if i < len(gestures) else [0, 0, 0, 0, 0]

        active_output_ds = keras.utils.timeseries_dataset_from_array(
            [gesture for _ in range(100)], None, sequence_length=1, batch_size=1)
        idle_output_ds = keras.utils.timeseries_dataset_from_array(
            [[0, 0, 0, 0, 0] for _ in range(100)], None, sequence_length=1, batch_size=1)

        idle_ds = tf.data.Dataset.zip((idle_input_ds, idle_output_ds))
        active_ds = tf.data.Dataset.zip((active_input_ds, active_output_ds))

# !FIXME: Will too much idle data cause problem? making it a unbalanced data
        if dataset_all is None:
            dataset_all = idle_ds
        else:
            dataset_all = dataset_all.concatenate(idle_ds)

        dataset_all = dataset_all.concatenate(active_ds)

    dataset_all = dataset_all.unbatch().batch(batch_size)

    return dataset_all
