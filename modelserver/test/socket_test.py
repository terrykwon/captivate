import websockets
import asyncio
import time

import json

async def web_recv():

    uri = "ws://0.0.0.0:8888"

    start_time = time.time()

    async with websockets.connect(uri) as websocket:

        await websocket.send("open_connection")

        while(1):
            message_recv = json.loads(await websocket.recv())

            if message_recv['tag'] == 'statistics':
                print("statistics_received")
                await consumer(message_recv)
                await websocket.send("close_connection")
                await websocket.close() # close connection
                break
            elif time.time() - start_time > 60:
                await websocket.send("end_process")
                print("send end process signal")

            else:
                await websocket.send(message_recv)
                await consumer(message_recv)
    print("end of connection(client)")            

async def consumer(message):
    print(message)
            
    

asyncio.get_event_loop().run_until_complete(web_recv())
