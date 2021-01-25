import os

import time
from multiprocessing import Queue, Process

import audiostream

from khaiii import KhaiiiApi

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/minkyungjeong/captivate/modelserver/default-demo-app-c4bc3-b67061a9c4b1.json"

url = 'rtmp://147.46.242.174:1935/captivate/test'

target_words = ["엄마","아빠","소파","침대","방","콩순이","멍멍이","베란다"]


class WordUsageHistory:
    
    def __init__(self, url, target_words):
        
        self.url = url
        self.queue = Queue()
        self.audial_process = Process(target=audiostream.run, args=(url, queue))
        
        self.target_words = target_words
        self.word_count = dict()
        self.word_count_final = dict()
        
        
        self.api = KhaiiiApi()
        
    def __enter__(self):
        self.audial_process.start()
        
        
    def get_word_count(self):
        
        self.word_count_final += self.word_count
        
        word_count_update = self.word_count
        
        self.word_count = {}
        
        return word_count_update
    
    def get_word_count_final(self):
        
        word_count_update = get_word_count()
        
        word_count_final_update = self.word_count_final
        
        self.word_count_final= {}
        
        
        return word_count_update, word_count_final_update
        
    
    def update_word_count(self):
        
        message = queue.get(block=True)

        transcript = message['transcript']

        transcript_processed = khaiii_lemmatization(transcript)

        for w in transcript_processed:
            if w in self.target_words:
                self.word_count[w] += 1
        
    def Khaiii_lemmatization(self, transcript_line):
        
        transcript_words = []
        
        try:
            line_pos = self.api.analyze(transcript_line)
            
            for w in line_pos:
                for m in w.morphs:
                        if m.tag in ['NNG', 'NR', 'MAG']:
                            transcript_words.append(m.lex)
                        elif m.tag in ['VV', 'VA']:
                            transcript_words.append(m.lex + '다')
                        elif m.tag in ['XR']:
                            transcript_words.append(m.lex+'하다')
                        else :
                            continue
        except Exception:
            pass
        
        return transcript_words

    
def run():
    
    Pwordusage = WordUsageHistory(url, target_words)
    
    with Pwordusage as p :
        start_time = time.time()
        
        while p.audial_process.is_alive():
            curr_time = time.time()
            
            if curr_time > start_time + 5:
                print(p.get_word_count())
                start_time = curr_time
            else :
                p.update_word_count()
        
        
        
if __name__ == "__main__":
    run()
                            
                            