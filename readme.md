# thehand-server

The server-side for the hand project that communicates with the web UI using websockets and web
requests.

To use this project, the `timeflux_eego` package needs to be manually installed. Please ensure that
you have also installed `PyBind11` and `msvc` before proceeding with the installation.

## TODOs

### Basics

- [x] Connect to the amplifier.

### Calibration session

- [x] Receive start and end requests from the web UI.
- [x] Preprocess data.
- [x] Finetune the model.

### Prediction session

- [x] Receive prediction start request from the web UI.
- [x] Transmit predicted data using websockets.

### Miscellaneous

- [ ] Improve the mock module.
- [ ] transmitting realtime emg signals
