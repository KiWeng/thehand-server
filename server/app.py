import asyncio
import logging
import time
import warnings

import aiohttp
import numpy as np
from aiohttp import web
from mne import filter

from fake import EegoDriver
from model.infer import EMGModel
from utils import DotTimer

# from utils import EegoDriver

log = logging.getLogger(__name__)

board = EegoDriver(sampling_rate=2000)
model = EMGModel()
dt = DotTimer()

# supress DeprecationWarning caused by mne filtering function
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

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

sample_frequency = 2000
freqs = (50.0, 100.0, 150.0, 200.0, 250, 300, 350, 400, 450, 500, 550, 600)
l_freq, h_freq = 8, 500


# begin of kalman
class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        """
        F: 状态转移矩阵
        H: 观测矩阵
        Q: 状态噪声协方差矩阵
        R: 观测噪声协方差矩阵
        P: 初始状态协方差矩阵
        x: 初始状态向量
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self):
        # 预测步骤
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # 更新步骤
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(len(self.x)) - np.dot(K, self.H), self.P)


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


async def recognition_handler(request):
    model_id = request.match_info['id']
    ws_current = web.WebSocketResponse()
    await ws_current.prepare(request)
    log.info(f"requesting model {model_id}")
    await ws_current.send_json({'action': 'connect', 'id': model_id})
    await asyncio.gather(close_ws(ws_current), recognition(ws_current, model_id))

    return ws_current


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

            filtered = filter.notch_filter((bipolar_data[:, :12] / stds).T, sample_frequency,
                                           freqs, filter_length="1000ms", trans_bandwidth=8.0,
                                           picks=range(12), verbose=False)
            filt = filter.create_filter(filtered, sample_frequency, l_freq, h_freq,
                                        verbose=False)
            filtered = filter._overlap_add_filter(filtered, filt,
                                                  picks=[i for i in range(12)]).T

            delay = 100

            normalized_data = np.expand_dims(filtered[:-delay][-800:, :], axis=0)
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
        web.get('/infer/{id}', recognition_handler),
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
