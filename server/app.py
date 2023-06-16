import asyncio
import logging

from aiohttp import web

import model
from utils import DotTimer
from utils import EegoDriver

log = logging.getLogger(__name__)

board = EegoDriver(sampling_rate=2000)
dt = DotTimer()


async def infer(request):
    model_id = request.match_info['id']
    ws_current = web.WebSocketResponse()
    await ws_current.prepare(request)
    log.info(f"requesting model {model_id}")
    EMGmodel = model.EMGModel(board)

    await ws_current.send_json({'action': 'connect', 'id': model_id})

    try:
        while True:
            await asyncio.sleep(0)
            prediction = EMGmodel.infer()
            await ws_current.send_json(
                {'action': 'sent', 'prediction': prediction.tolist()}
            )
            dt.dot()
            print(f'{dt.avg}--time elapsed')

    except OSError:
        log.info(f"stop inferring model {model_id}")

    return ws_current


async def calibration(request):  # TODO
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

    return ws_current


async def init_app():
    app = web.Application()

    app['websockets'] = {}
    app.on_shutdown.append(shutdown)

    app.add_routes([
        web.get('/infer/{id}', infer),
        web.get('/calibration', calibration)
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
