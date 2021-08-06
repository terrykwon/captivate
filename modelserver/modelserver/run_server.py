from collections import defaultdict
import os
import psutil

# from multiprocessing import Process, Queue
from torch import multiprocessing
from torch.multiprocessing import Process, Queue

import sys

from websockets.exceptions import ConnectionClosedOK
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import asyncio
import websockets
import json

from modelserver import server


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

statistic_num = 8


url = 'rtmp://video:1935/captivate/test'

data_queue = Queue()

statistic_word = defaultdict(int)
statistic_object = defaultdict(int)

server_process = None

def get_list(target_dict):
    target_list = []
    for k, v in target_dict.items():
        target_list.append({'name':k, 'count':v})

    target_list = target_list[:8]

    return target_list

async def consumer(message, websocket):

    if message == "open_connection":
        killtree(os.getpid(), False)

        server_process = Process(target=server.start, args=(url, data_queue, True)) 

        server_process.start()

    elif message == "end_process":
        statistics = {
            'tag' : 'statistics',
            'wordStatistic' : get_list(statistic_word),
            'objectStatistic' : get_list(statistic_object)
        }
        
        print("send statistics!!!")
        await websocket.send(json.dumps(statistics))

        statistic_word.clear()
        statistic_object.clear()

    elif message == "close_connection":
        await websocket.close()
        print("websocket connection closed!!!")
        killtree(os.getpid(), False)
        
    else:
        print("data received from client")


async def consumer_handler(websocket, path):

    while not websocket.closed :
        message = await websocket.recv()
        await consumer(message, websocket)
    

async def producer():
    if not data_queue.empty():
        message = data_queue.get(block=True, timeout=1.0)

        if message['tag'] == 'target_words':
            target_word_list = message['words']
            for word in target_word_list:
                statistic_word[word] += 1
            
        elif message['tag'] == 'recommendation' :
            attended_object = message['recommendations'][0]['object']
            statistic_object[attended_object] += 1
        
        return json.dumps(message)
    else:
        await asyncio.sleep(1)
        return False

async def producer_handler(websocket, path):
    while True:
        message = await producer()
        if message:
            await websocket.send(message)
                
    

        
    
async def websocket_handler(websocket, path):
    try:
        consumer_task = asyncio.ensure_future(
            consumer_handler(websocket, path))
        producer_task = asyncio.ensure_future(
            producer_handler(websocket, path))
        done, pending = await asyncio.wait(
            [consumer_task, producer_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

    except Exception as excp:
        print(type(excp))
        print(excp.args)
        print(excp)
        killtree(os.getpid(), False)
    
    # for task in pending:
    #     task.cancel()


def killtree(process_pid, including_parent):
    parent = psutil.Process(process_pid)
    for child in parent.children(recursive=False):
        killtree(child.pid, True)
    
    if including_parent:
        parent.kill()



if __name__ == '__main__': 
    
    ## for server test 
    # server_process = Process(target=start, 
    #             args=(url, data_queue, True))
    
    # server_process.start()
    
    # while server_process.is_alive():
    #     if not data_queue.empty():
    #         result = data_queue.get(block=True)
    #         print(result)
    #         print('\n')

    ## for websocket
    websocket_server = websockets.serve(websocket_handler, '0.0.0.0', 8888, ping_interval = None)

    asyncio.get_event_loop().run_until_complete(websocket_server)
    asyncio.get_event_loop().run_forever()
    
    
    
