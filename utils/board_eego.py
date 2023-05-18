import time

import eego
import numpy as np


class EegoDriver:

    def __init__(self,
                 sampling_rate=2000,
                 reference_channels=None,
                 reference_range=1,
                 bipolar_channels=None,
                 bipolar_range=4,
                 amplifier_index=0,
                 ):
        super().__init__()
        self._factory = eego.glue.factory(eego.sdk.default_dll(), None)
        self._amplifier = None

        retries = 3
        while retries > 0:
            retries -= 1
            try:
                self._amplifier = self._factory.amplifiers[amplifier_index]
            except IndexError:
                print(f'Amplifier {amplifier_index} not found, retrying...')
                time.sleep(1)
            if self._amplifier:
                print('Connected to amplifier')
                break
        if not self._amplifier:
            print('Could not find EEG amplifier, is it connected and on?')
            raise ValueError('Could not initialize EEG amplifier')
        self._ref_config = self._amplifier.get_default_config('reference',
                                                              names=reference_channels,
                                                              signal_range=reference_range)
        self._bip_config = self._amplifier.get_default_config('bipolar',
                                                              names=bipolar_channels,
                                                              signal_range=bipolar_range)

        if sampling_rate not in self._amplifier.sampling_rates:
            raise ValueError(
                f'Unsupported sampling rate {sampling_rate} by 'f'{self._amplifier}')
            # TODO: amplifier repr or str
        self._rate = sampling_rate

        print(f'Masks are {self._bip_config.mask} {self._ref_config.mask}')

        self._stream = self._amplifier.open_eeg_stream(self._rate,
                                                       self._ref_config.range,
                                                       self._bip_config.range,
                                                       self._ref_config.mask,
                                                       self._bip_config.mask)

        self._start_timestamp = None
        self._reference_ts = None
        self._sample_count = None
        self.eeg_data = None
        self.bip_data = None

        print('Eeego amplifier connected %s' % self._amplifier)

    def get_data(self):
        # The first time, drop all samples that might have been captured
        # between the initialization and the first time this is called
        if self._sample_count is None:
            buffer = self._stream.get_data()
            n_samples, n_channels = buffer.shape
            print(f'Dropped a total of {n_samples} samples of data'
                  f' between driver initialization and first node update')
            self._sample_count = 0

        try:
            buffer = self._stream.get_data()
        except RuntimeError as ex:
            print(f'Eego SDK gave runtime error ({ex}), resuming the driver acquisition...')
            return
        n_samples, n_channels = buffer.shape
        if n_samples <= 0:
            print('No data yet...')
            return

        data = np.fromiter(buffer, dtype=np.float64).reshape(-1, n_channels)
        del buffer

        # account for read data for starting timestamp
        if self._sample_count == 0 and n_samples > 0:
            self._start_timestamp = (
                    np.datetime64(int(time.time() * 1e6), 'us') -
                    # Adjust for the read samples
                    int(1e6 * n_samples / self._rate)
            )
            self._reference_ts = self._start_timestamp

        # sample counting to calculate drift
        self._sample_count += n_samples
        elapsed_seconds = (
                (np.datetime64(int(time.time() * 1e6), 'us') - self._reference_ts) /
                np.timedelta64(1, 's')
        )
        n_expected = int(np.round(elapsed_seconds * self._rate))
        print('Read samples=%d, elapsed_seconds=%f. '
              'Expected=%d Real=%d Diff=%d (%.3f sec)'
              % (n_samples, elapsed_seconds,
                 n_expected, self._sample_count,
                 n_expected - self._sample_count,
                 (n_expected - self._sample_count) / self._rate))

        # Manage timestamps
        # For this node, we are trusting the device clock and setting the
        # timestamps from the sample number and sampling rate
        timestamps = (
                self._start_timestamp +
                (np.arange(n_samples + 1) * 1e6 / self._rate).astype('timedelta64[us]')
        )
        self._start_timestamp = timestamps[-1]

        eeg_col_idx = np.r_[
            np.arange(len(self._ref_config.channels)), [-2, -1]]
        self.eeg_data = data[:, eeg_col_idx]

        bip_col_idx = np.r_[
            np.arange(len(self._bip_config.channels)) + len(self._ref_config.channels), [-2, -1]]
        self.bip_data = data[:, bip_col_idx]

        return self.eeg_data, self.bip_data


if __name__ == "__main__":
    dri = EegoDriver()
    dri.get_data()
    bip_data = dri.bip_data
    while bip_data is None:
        dri.get_data()
        bip_data = dri.bip_data
    print(bip_data.shape)
    print(bip_data)

    time.sleep(1)
    dri.get_data()
    bip_data = dri.bip_data
    print(bip_data.shape)  # (2032, 26), for sampling rate@2000hz
    print(bip_data)
