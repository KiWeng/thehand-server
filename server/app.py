import asyncio
import logging
import time

import numpy as np
from aiohttp import web
from mne import filter

from utils import EegoDriver
# from fake import EegoDriver
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


async def infer(request):
    model_id = request.match_info['id']
    ws_current = web.WebSocketResponse()
    await ws_current.prepare(request)
    log.info(f"requesting model {model_id}")

    sample_len = 6000

    def step(driver):
        data = driver.get_data_of_size(sample_len)
        bipolar_data = data[1]
        while bipolar_data.shape[0] < sample_len:
            time.sleep(0.2)
            data = driver.get_data_of_size(sample_len)
            bipolar_data = data[1]

        filtered = filter.notch_filter((bipolar_data[:, :12] / stds).T, sample_frequency, freqs,
                                       picks=range(12), verbose=False)
        filt = filter.create_filter(filtered, sample_frequency, l_freq, h_freq, verbose=False)
        filtered = filter._overlap_add_filter(filtered, filt, picks=[i for i in range(12)]).T

        print(filtered.shape)
        delay = 1000
        print(filtered[:-delay][-800:, :].std(axis=0))

        # print(normalized_data[-1:, :])

        # normalized_data = np.expand_dims(filtered[:-1000][-800:, :], axis=0)
        normalized_data = np.expand_dims(filtered[:-delay][-800:, :], axis=0)
        '''
        TODO: There are some problems:
        1. Some times some channels are sinusoidal so it will create large spikes
         at the start&end of the signal, so head and tail must be cut
        2. A overlong segment is needed to have a good filtered result
        '''
        prediction = model.predict(normalized_data)  # (samples, channels)

        filtered_pred = []
        for i in range(len(prediction[0])):
            kfs[i].predict()
            kfs[i].update(prediction[0][i])
            filtered_pred.append(kfs[i].x[0])


        print(prediction)
        print([filtered_pred])

        return np.asarray([filtered_pred])

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
