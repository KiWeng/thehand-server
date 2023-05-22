import asyncio
import logging
import time

import numpy as np
from aiohttp import web
from scipy import signal

# from utils import EegoDriver
from fake import EegoDriver
from model.infer import EMGModel
from utils import DotTimer

log = logging.getLogger(__name__)

board = EegoDriver(sampling_rate=2000)
model = EMGModel()
dt = DotTimer()

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


async def infer(request):
    model_id = request.match_info['id']
    ws_current = web.WebSocketResponse()
    await ws_current.prepare(request)
    log.info(f"requesting model {model_id}")

    def step(driver):
        data = driver.get_data_of_size(800)
        bipolar_data = data[1]
        while bipolar_data.shape[0] < 800:
            time.sleep(0.2)
            data = driver.get_data_of_size(800)
            bipolar_data = data[1]

        filtered_data = signal.filtfilt(notch_b, notch_a, bipolar_data[:, :12], axis=0)
        filtered_data = signal.filtfilt(band_b, band_a, filtered_data, axis=0)

        print(filtered_data.std(axis=0))

        normalized_data = filtered_data / stds

        print(normalized_data.std(axis=0))

        # print(normalized_data[-1:, :])

        normalized_data = np.expand_dims(normalized_data, axis=0)
        prediction = model.predict(normalized_data)  # (samples, channels)
        print(prediction)

        return prediction

    await ws_current.send_json({'action': 'connect', 'id': model_id})

    try:
        while True:
            # too slow in test
            # if board.get_board_data_count() < 8 * 800:
            #     continue

            # elegant, but slower, fuck
            # prediction = await asyncio.to_thread(infer_step, board)
            prediction = step(board)
            await asyncio.sleep(0)
            await ws_current.send_json(
                {'action': 'sent', 'prediction': prediction.tolist()}
            )
            dt.dot()
            print(f'{dt.avg}--time elapsed')

    except OSError:
        log.info(f"stop inferring model {model_id}")

    return ws_current


async def recording(request):  # TODO
    ws_current = web.WebSocketResponse()
    await ws_current.prepare(request)
    log.info(f"requesting model ")

    await ws_current.send_json({'action': 'connect', 'id': ' model_id'})

    # try:
    while not ws_current.closed:
        await asyncio.sleep(1)
        await ws_current.send_json(
            {'action': 'sent', 'id': ' model_id'}
        )
    # except OSError:
    #     log.info(f"stop inferring model ")

    return ws_current


async def init_app():
    app = web.Application()

    app['websockets'] = {}
    app.on_shutdown.append(shutdown)

    app.add_routes([
        web.get('/infer/{id}', infer),
        web.get('/', recording)
    ])

    # TODO

    return app


async def shutdown(app):
    # TODO
    for ws in app['websockets'].values():
        await ws.close()
    app['websockets'].clear()


def main():
    logging.basicConfig(level=logging.DEBUG)
    app = init_app()
    web.run_app(app, port=8081)
