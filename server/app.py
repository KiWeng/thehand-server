import asyncio
import json
import logging
import os
import time

import aiohttp
import aiohttp_cors
import numpy as np
from aiohttp import web

# from fake import EegoDriver
from model.model import EMGModel
from utils import DotTimer, KalmanFilter, filter_data, make_calibration_ds
from utils import EegoDriver
from utils.record import record

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


async def handle_message(ws_current):
    start_time, stop_time = 0, 0
    async for msg in ws_current:
        if msg.type == aiohttp.WSMsgType.TEXT:
            data = json.loads(msg.data)
            if data['type'] == 'stop':
                stop_time = int(data['stop_time'])
                start_time = int(data['start_time'])
                return start_time, stop_time
        elif msg.type == aiohttp.WSMsgType.ERROR:
            print('ws connection closed with exception %s' %
                  ws_current.exception())
    return start_time, stop_time


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

            print(stds)

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
            model.reload_model(f"../assets/saved_model/{model_id}")
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


async def preprocess_data(ws_current, duration=30):  # TODO
    await asyncio.sleep(0)
    await ws_current.send_json(
        {'action': "start recording"}
    )

    bipolar_data, last_record_time = record(board, duration)  # TODO
    np.savetxt(f"../tmp/calibration_record_{time.time()}.csv", bipolar_data, delimiter=",")

    # print(bipolar_data.shape)
    bipolar_data = bipolar_data[:, :12]
    current_stds = bipolar_data.std(axis=0)

    filtered_data = filter_data((bipolar_data / current_stds).T).T
    await asyncio.sleep(0)
    await ws_current.send_json(
        {'action': "start calibration"}
    )
    return filtered_data, current_stds, last_record_time


async def calibrate(ws_current, filtered_data, gestures, stop_time, last_record_time,
                    model_dst="../assets/saved_model/tmp"):
    # TODO
    ds = make_calibration_ds(filtered_data, gestures)
    model.calibrate(ds, model_dst)

    await asyncio.sleep(0)
    await ws_current.send_json(
        {'action': "calibration finished"}
    )


async def calibration_handler(request):
    model_id = request.match_info['id']
    ws_current = web.WebSocketResponse()
    await ws_current.prepare(request)
    await ws_current.send_json({'action': 'connected', 'id': model_id})
    """
    This maybe not the most elegant way to collect data, it can be improved by:
    - Use web requests to start/stop the data collection process
    - Use websocket
    """
    """
    TODO:
    The client will have to wait a long time to get a result, this may be a problem
    """

    # TODO: this should be acquired from the client
    gestures = [
        [16, 0, 0, 0, 0],
        [0, 16, 0, 0, 0],
        [0, 0, 16, 0, 0],
        [0, 0, 0, 16, 0],
        [0, 0, 0, 0, 16],
        [16, 16, 16, 16, 16],
        [16, 0, 0, 16, 16],
        [16, 0, 0, 0, 0],
        [0, 16, 0, 0, 0],
        [0, 0, 16, 0, 0],
        [0, 0, 0, 16, 0],
        [0, 0, 0, 0, 16],
        [16, 16, 16, 16, 16],
        [16, 0, 0, 16, 16],
        [16, 0, 0, 0, 0],
        [0, 16, 0, 0, 0],
        [0, 0, 16, 0, 0],
        [0, 0, 0, 16, 0],
        [0, 0, 0, 0, 16],
        [16, 16, 16, 16, 16],
        [16, 0, 0, 16, 16],
        [16, 0, 0, 0, 0],
        [0, 16, 0, 0, 0],
        [0, 0, 16, 0, 0],
        [0, 0, 0, 16, 0],
        [0, 0, 0, 0, 16],
        [16, 16, 16, 16, 16],
        [16, 0, 0, 16, 16],
        [16, 0, 0, 0, 0],
        [0, 16, 0, 0, 0],
        [0, 0, 16, 0, 0],
        [0, 0, 0, 16, 0],
        [0, 0, 0, 0, 16],
        [16, 16, 16, 16, 16],
        [16, 0, 0, 16, 16],
    ]

    results = await asyncio.gather(handle_message(ws_current),
                                   preprocess_data(ws_current, len(gestures) * 5))
    start_time, stop_time = results[0]
    global stds
    filtered_data, stds, last_record_time = results[1]

    print(f'new std: {stds}')
    print(f'client stop time: {stop_time}\nserver stop time: {last_record_time}')
    # Presumably, server stops later
    end_time_diff_ms = int(last_record_time * 1000) - int(stop_time)
    print(f'start time: {start_time}\nstop time: {stop_time}\nend time diff: {end_time_diff_ms}')
    print(filtered_data.shape)

    fd_shape = filtered_data.shape
    filtered_data = filtered_data[:fd_shape[0] - end_time_diff_ms * 2, :]
    fd_shape = filtered_data.shape
    print(filtered_data.shape)

    print(2 * (stop_time - start_time) - fd_shape[0])

    filtered_data = np.pad(
        filtered_data,
        ((2 * (stop_time - start_time) - fd_shape[0], 0), (0, 0)),
        'mean')
    print(filtered_data.shape)

    np.savetxt(f"../tmp/calibration_record_{time.time()}_filtered.csv", filtered_data, delimiter=",")
    np.savetxt(f"../tmp/calibration_record_{time.time()}_stds.csv", stds, delimiter=",")

    # print(results)
    await asyncio.gather(close_ws(ws_current),
                         calibrate(ws_current, filtered_data, gestures, stop_time,
                                   last_record_time))

    return ws_current


async def list_models(request):
    model_list = os.listdir("../assets/saved_model/")
    response_data = {'models': model_list}
    return web.Response(text=json.dumps(response_data), status=200, content_type='application/json')


async def init_app():
    app = web.Application()

    app['websockets'] = {}
    app.on_shutdown.append(shutdown)

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*"
        )
    })

    app.add_routes([
        web.get('/infer/{id}', recognition_handler),
        web.get('/models/', list_models),
        web.get('/calibration/{id}', calibration_handler),
    ])

    for route in list(app.router.routes()):
        cors.add(route)

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
