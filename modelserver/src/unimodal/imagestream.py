from threading import Thread
from queue import Queue
from collections import deque
# from asyncio import Lock
from threading import Lock
import time
import cv2


class ImageStream:
    """ Maintains a queue of the N most recent frames from a live video stream.
        Older frames are discarded.
    """

    def __init__(self, path, buffer_length=16):
        self.video_capture = cv2.VideoCapture(path)

        if not self.video_capture.isOpened():
            print('Unable to open VideoCapture')
            return

        print('VideoCaptured opened at', path)

        # self.buffer = Queue(maxsize=buffer_length)
        self.buffer = deque(maxlen=buffer_length)
        self.running = False # flag
        self.buffer_length = buffer_length

        self.lock = Lock()

    def start(self):
        thread = Thread(target=self.update, args=())
        thread.daemon = True
        self.running = True
        thread.start()

        return self

    def stop(self):
        self.running = False

    def update(self):
        # print('update')
        while self.running:
            self.lock.acquire()
            # print('acquired lock in update')
            is_read, frame = self.video_capture.read()
            # print('is_read', is_read)

            if not is_read:
                print('is not read, stopping')
                self.running = False
                self.lock.release()
                return

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.buffer.append(frame)
            self.lock.release()

            time.sleep(0.01) # give time for dump() to acquire lock

    def dump(self):
        ''' Empties the current queue.
            This should be blocking.

            Or actually it doesn't need to be blocking, since
            all we have to do is read the current contents of the buffer, not
            write or modify it.
        '''
        frames = []

        while len(self.buffer) == 0:
            time.sleep(0.01) # this typically only happens for the first frame, before inference

        self.lock.acquire()
        while len(self.buffer) > 0:
            frames.append(self.buffer.popleft())
        self.lock.release()

        return frames