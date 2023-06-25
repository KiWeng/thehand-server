import asyncio
import logging
import time

import aiohttp
import numpy as np
from aiohttp import web

from fake import EegoDriver
from model.model import EMGModel
from utils import DotTimer, KalmanFilter, filter_data, make_calibration_ds
from utils.record import record

# from utils import EegoDriver

log = logging.getLogger(__name__)

board = EegoDriver(sampling_rate=2000)
model = EMGModel("../assets/saved_model/tmp")
dt = DotTimer()

# TODO this can be after calibration
stds = [
    0.000034,
    0.000019,
    0.000013,
    0.300000,
    0.000012,
    0.000014,
    0.000031,
    0.000028,
    0.000036,
    0.000028,
    0.000051,
    0.000047,
]  # std value exclude extreme is 0.00003

# 定义观测矩阵和状态转移矩阵
H = np.array([[1]])
F = np.array([[1]])

# 定义初始状态向量和协方差矩阵
x = np.array([0])
P = np.eye(1)

# 定义噪声协方差矩阵
Q = np.array([[1]])
R = np.array([[1]])

# end of kalman
kfs = [KalmanFilter(F=F, H=H, Q=Q, R=R, P=P, x=x) for i in range(5)]


async def close_ws(ws_current):
    async for msg in ws_current:
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == 'close':
                await ws_current.close()
                await asyncio.sleep(0)
            else:
                await ws_current.send_str(msg.data + '/answer')
        elif msg.type == aiohttp.WSMsgType.ERROR:
            print('ws connection closed with exception %s' %
                  ws_current.exception())


async def recognition_handler(request):
    model_id = request.match_info['id']
    ws_current = web.WebSocketResponse()
    await ws_current.prepare(request)
    log.info(f"requesting model {model_id}")
    await ws_current.send_json({'action': 'connect', 'id': model_id})
    await asyncio.gather(close_ws(ws_current), recognition(ws_current, model_id))
    return ws_current


async def recognition(ws_current, model_id):
    sample_len = 2000

    try:
        while not ws_current.closed:
            data = board.get_data_of_size(sample_len)
            bipolar_data = data[1]
            while bipolar_data.shape[0] < sample_len:
                time.sleep(0.2)
                data = board.get_data_of_size(sample_len)
                bipolar_data = data[1]
            filtered = filter_data((bipolar_data[:, :12] / stds).T)
            delay = 100
            normalized_data = np.expand_dims(filtered.T[:-delay][-800:, :], axis=0)
            '''
            TODO: There are some problems:
            1. Some times some channels are sinusoidal so it will create large spikes
             at the start&end of the signal, so head and tail must be cut
            2. A overlong segment is needed to have a good filtered result
            '''
            '''
            20230615
            The problem still remains, by changing the trans_bandwidth & filter_length the lag have been
            greatly reduced but still remains (delay can be set to 0 but will the quality of the
            filtered data will be worsen)
            Things that affect filtered data quality:
                change the filtered_slice length (longer is better)
                change the filter_length&trans_bandwidth in notch_filter parameters
                padding
            '''
            prediction = model.predict(normalized_data)  # (samples, channels)
            filtered_pred = []
            for i in range(len(prediction[0])):
                kfs[i].predict()
                kfs[i].update(prediction[0][i])
                filtered_pred.append(kfs[i].x[0])
            filtered_pred = np.asarray([filtered_pred])

            '''
            prediction = await asyncio.to_thread(run_task, params)
            elegant, but slower, fuck
            '''
            await asyncio.sleep(0)
            await ws_current.send_json(
                {'action': 'sent', 'prediction': filtered_pred.tolist()}
            )
            dt.dot()
    finally:
        return


def calibration(duration=30, model_dst="../assets/saved_model/tmp"):  # TODO
    print("record started")
    bipolar_data = record(board, duration)[:, :12]  # TODO
    current_stds = bipolar_data.std(axis=0)

    filtered_data = filter_data((bipolar_data / current_stds).T).T
    ds = make_calibration_ds(filtered_data)
    model.calibrate(ds, model_dst)
    return current_stds


async def calibration_handler(request):
    data = await request.post()
    """
    This maybe not the most elegant way to collect data, it can be improved by:
    - Use web requests to start/stop the data collection process
    - Use websocket
    """
    duration = int(data['duration'])
    stds = await asyncio.to_thread(calibration, duration)
    return web.json_response({
        'response': 'ok'
    })


async def model_id():
    return web.json_response({
        'response': 'ok'
    })


async def init_app():
    app = web.Application()

    app['websockets'] = {}
    app.on_shutdown.append(shutdown)

    app.add_routes([
        web.get('/infer/{id}', recognition_handler),
        web.get('/model/{id}', model_id),
        web.post('/calibration/start', calibration_handler),
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
