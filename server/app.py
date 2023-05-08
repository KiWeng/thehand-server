from aiohttp import web

from views import index


async def init_app():
    app = web.Application()

    app['websockets'] = {}

    app.on_shutdown.append(shutdown)

    app.router.add_get('/', index)

    return app


async def shutdown(app):
    for ws in app['websockets'].values():
        await ws.close()
    app['websockets'].clear()


def main():
    app = init_app()
    web.run_app(app)
