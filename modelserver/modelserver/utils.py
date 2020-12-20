import time


class Fps:
    """ Class to calculate the FPS over a fixed-length window.
    """
    def __init__(self, window=10):
        self.window = window
        self.counter = 0
        self.prev_time = time.time() # first FPS value is inaccurate
        self._fps = 0.0
    
    def count(self):
        self.counter += 1
        if self.counter % self.window == 0:
            self.counter = 0
            cur_time = time.time()
            self._fps = self.window / (cur_time - self.prev_time)
            self.prev_time = cur_time
        
        return self._fps
