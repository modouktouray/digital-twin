import asyncio
from mangum import Mangum
from server import app

# FIX for Python 3.12+
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

handler = Mangum(app)
