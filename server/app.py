import asyncio
import logging

import aiohttp
from aiohttp import web

import model
from fake import EegoDriver
from utils import DotTimer

# from utils import EegoDriver

log = logging.getLogger(__name__)

board = EegoDriver(sampling_rate=2000)
dt = DotTimer()


async def recognition_handler(request):
    model_id = request.match_info['id']
    ws_current = web.WebSocketResponse()
    await ws_current.prepare(request)
    log.info(f"requesting model {model_id}")

    await ws_current.send_json({'action': 'connect', 'id': model_id})
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
    try:
        while not ws_current.closed:
            EMGModel = model.EMGModel(board)
            prediction = EMGModel.infer()

            '''
            prediction = await asyncio.to_thread(run_task, params)
            elegant, but slower, fuck
            '''
            await asyncio.sleep(0)
            await ws_current.send_json(
                {'action': 'sent', 'prediction': prediction}
            )
            dt.dot()
    finally:
        return


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
        web.get('/infer/{id}', recognition_handler),
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
