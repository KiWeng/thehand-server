import asyncio
import logging
import time
from random import random

import numpy as np
from aiohttp import web

from models.infer import load_model
from utils import load_board

log = logging.getLogger(__name__)

board = load_board(fake=False)
model = load_model()


async def infer(request):
    model_id = request.match_info['id']
    ws_current = web.WebSocketResponse()
    await ws_current.prepare(request)
    log.info(f"requesting model {model_id}")

    await ws_current.send_json({'action': 'connect', 'id': model_id})

    prev_time = time.time_ns()

    try:
        while True:
            # too slow in test
            # if board.get_board_data_count() < 8 * 800:
            #     continue

            current_time = time.time_ns()
            print(f"{((current_time - prev_time) / 1e6)} start")
            prev_time = current_time

            fake_data = board.get_current_board_data(8 * 800)[:12, ::8]
            fake_data = fake_data[:, -800:]
            fake_data = np.expand_dims(fake_data.T, axis=0)

            prediction = model.predict(fake_data)
            await asyncio.sleep(0)
            ''' TODO:
            This is only a workaround, see p552 in Fluent Python (chinese) for detailed information
            '''
            await ws_current.send_json(
                {'action': 'sent', 'prediction': prediction.tolist()}
            )
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
            ''' TODO:
            This is only a workaround, see p552 in Fluent Python (chinese) for detailed information
            '''
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
    await app.ws.close()


async def shutdown(app):
    for ws in app['websockets'].values():
        await ws.close()
    app['websockets'].clear()


def main():
    logging.basicConfig(level=logging.DEBUG)
    app = init_app()
    web.run_app(app, port=8081)
