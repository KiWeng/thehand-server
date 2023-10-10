import numpy as np


def replace_outliers(data, threshold=2):
    num_channels = data.shape[1]

    modified = np.empty_like(data)
    # Iterate over each channel
    for channel in range(num_channels):
        channel_data = data[:, channel]
        mean = np.mean(channel_data)
        channel_data[abs(channel_data) > 10] = mean
        modified[:, channel] = channel_data

    return modified
