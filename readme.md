# thehand-server

The server-side for the hand project that communicates with the web UI using websockets and web
requests.

To use this project, the `timeflux_eego` package needs to be manually installed. Please ensure that
you have also installed `PyBind11` and `msvc` before proceeding with the installation.

## TODOs

### Basics

- [ ] Connect to the amplifier.

### Calibration session

- [ ] Receive start and end requests from the web UI.
- [ ] Preprocess data.
- [ ] Finetune the model.

### Prediction session

- [ ] Receive prediction start request from the web UI.
- [ ] Transmit predicted data at 50Hz using websockets.

### Miscellaneous

- [ ] Improve the mock module.
