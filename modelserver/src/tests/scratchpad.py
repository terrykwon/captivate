import asyncio
import sys
import signal
from asyncio.subprocess import PIPE, STDOUT

from imagestream import ImageStream
from modelserver import MultimodalMonitor
from wrappers.speech_recognizer import SpeechRecognizer

import cv2
import ffmpeg
# from audiostream import AudioStream

''' main event loop '''

url = 'rtmp:/video:1935/captivate/test'

# run decoding subprocesses in background
# imagestream should only yield the most recent images, and discard the others
# imagestream = ImageStream()

# audiostream should yield ALL audio frames, as close to real-time as possible
# audiostream = AudioStream()


# video_capture = cv2.VideoCapture(url)
# monitor = MultimodalMonitor(None)

# async def produce():
#     read, frame = video_capture.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     return monitor.predict_single_frame(frame, None)

# async def main():
#     c = 0
#     while 1:
#         outputs = await produce()
#         print(outputs)


async def run_decoder():
    timeout = 5.0 # seconds
    speech_recognizer = SpeechRecognizer()
    # speech_recognizer.start_decoder_subprocess(url)# asyncio.run(main())

    args = (
        ffmpeg
        .input(url)
        .output('pipe:', format='s16le', acodec='pcm_s16le')
        .compile()
    )

    ffmpeg_process = await asyncio.create_subprocess_exec(
        *args, stdout=PIPE
    )

    while 1:
        try:
            frame = await asyncio.wait_for(
                    ffmpeg_process.stdout.read(8192), 
                    timeout=timeout)
        except asyncio.TimeoutError:
            print('Connection time out!')
            ffmpeg_process.send_signal(signal.SIGINT) # gracefully stop
        else:
            if not frame: # EOF
                print('EOF')
                break

            continue # 
            # todo: create request with frame
        ffmpeg_process.send_signal(signal.SIGINT) # gracefully stop
        break

    return await ffmpeg_process.wait() # wait for the child process to exit


loop = asyncio.get_event_loop()
loop.run_until_complete(run_decoder())
loop.close()