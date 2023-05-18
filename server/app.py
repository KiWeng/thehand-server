import asyncio
import logging
from random import random

import numpy as np
from aiohttp import web

from model.infer import EMGModel
from utils import load_board, DotTimer

log = logging.getLogger(__name__)

board = load_board(fake=False)
model = EMGModel()
dt = DotTimer()


async def infer(request):
    model_id = request.match_info['id']
    ws_current = web.WebSocketResponse()
    await ws_current.prepare(request)
    log.info(f"requesting model {model_id}")

    def step(device):
        data = device.get_current_board_data(8 * 800)[:12, ::8]
        data = data[:, -800:]
        data = np.expand_dims(data.T, axis=0)
        return model.predict(data)

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


async def infer_fake(request):
    model_id = request.match_info['id']
    ws_current = web.WebSocketResponse()
    await ws_current.prepare(request)
    log.info(f"requesting model {model_id}")

    await ws_current.send_json({'action': 'connect', 'id': model_id})

    try:
        while True:
            await asyncio.sleep(0.1)
            await ws_current.send_json(
                {'action': 'sent', 'prediction': [
                    [random() * 15, random() * 15, random() * 15, random() * 15, random() * 15]]}
            )
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
        # web.get('/infer/{id}', infer_fake),
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
