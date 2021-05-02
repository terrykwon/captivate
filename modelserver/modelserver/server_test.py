import os

# from multiprocessing import Process, Lock, Barrier, Queue
from torch import multiprocessing
from torch.multiprocessing import Process, Lock, Barrier, Queue

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from modelserver.unimodal import audial_infinite, visual
from modelserver.visualization.visualizer import Visualizer

import time
import math

import numpy as np
import heapq

from khaiii import KhaiiiApi

from collections import defaultdict
import re


import asyncio
import websockets
import json

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/workspace/modelserver/default-demo-app-c4bc3-b67061a9c4b1.json"


url = 'rtmp://video:1935/captivate/test'
# url ='/workspace/modelserver/test.flv'

data_queue = Queue()



class ModelServer:
    ''' The "server" is a main process that starts two additional processes: the
        'visual' process and the 'audial' process. Each process runs streaming
        inference on the decoded outputs of a video stream, which is done by
        starting decoder subprocesses.

        The inference outputs of the two processes are put into a `Queue`, where
        they are consumed by the main process.

        TODO:
          - Graceful shutdown. Currently one has to re-run the whole pipeline to
            start inference on a new video, which is very annoying. This can be
            done with some form of locking? Since the models have to be
            initialized in their respective thread.

          - Running the pipeline on a local video file and saving the output,
            for demo purposes.
    '''

    def __init__(self, stream_url, result_queue):
        # this method must be only called once
        # multiprocessing.set_start_method('spawn') # CUDA doesn't support fork
        # but doesn't matter if CUDA is initialized on a single process
        self.result_queue = result_queue
        
        self.stream_url = stream_url
        self.queue = Queue() # thread-safe


        self.visual_process_1 = Process(target=visual.run, 
                args=(self.stream_url+'1', self.queue, None))

        # self.visual_process_2 = Process(target=visual.run, 
        #         args=(self.stream_url+'2', self.queue, None))
        
        # self.visual_process_3 = Process(target=visual.run, 
        #         args=(self.stream_url+'3', self.queue, None))
        
        self.audial_process = Process(target=audial_infinite.run, 
                args=(self.stream_url+'1', self.queue, None))

        self.visualizer = Visualizer()

        self.Khaiii_api = KhaiiiApi()


        self.objects = ['공','신발','숟가락','그릇','포크','버스','자전거','물고기','강아지','고양이','거울','칫솔','양말','선물','꽃','텔레비전']
        self.visual_classes = {
            'ball' : '공',
            'dog' : '강아지',
            'cat' : '고양이',
            'shoe' : '신발',
            'spoon' : '숟가락',
            'bowl' : '그릇',
            'fork' : '포크',
            'bus' : '버스',
            'tv' : '텔레비전',
            'bicycle' : '자전거',
            'fish' : '물고기',
            'mirror' : '거울',
            'toothbrush' : '칫솔',
            'sock' : '양말',
            'gift' : '선물',
            'flower' : '꽃'
        }
 
        self.context = {obj : 1/len(self.objects) for obj in self.objects}
        self.candidates = {
            '공' : {'치다': 1, '던지다' : 1, '탁탁' : 1,'때리다':1,'박수치다':1, '올라가다':1, '발':1,'망치':1, '테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,},
            '신발' : {'신다': 1,'옷': 1,'바지': 1, '운동화':1, '입다':1,'양말':1,'옹기종기':1,'구두':1, '테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,},
            '숟가락' : {'젓가락': 1 , '포크' : 1, '접시': 1,'꼭꼭':1, '딸가닥':1,'컵':1,'밥':1,'동동':1, '테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,},
            '그릇' : {'접시' : 1, '밥' : 1,'김치' : 1,'라면' : 1, '뚝딱뚝딱' : 1, '젓가락' : 1,'수박' : 1,'전자레인지' : 1, '테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,},
            '포크' : {'숟가락' : 1, '가위' : 1, '젓가락' : 1, '딸가닥' : 1, '빗자루' : 1, '쟁반' : 1, '자르다' : 1, '뚝딱뚝딱' : 1 ,'테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,},
            '버스' : {'기차' : 1, '택시' : 1, '타다' : 1, '내리다' : 1, '공항' : 1, '헬리콥터' : 1, '아슬아슬' : 1, '뚜벅뚜벅' : 1,'테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,},
            '자전거' : {'씽씽' : 1, '오토바이' : 1, '타다' : 1, '쌩쌩' : 1, '버스' : 1, '걷다' : 1, '운동화' : 1, '다니다' : 1,'테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,} ,
            '물고기' : {'바다' : 1,'토끼' : 1,'미끄럼틀' : 1, '생선' : 1, '거북이' : 1, '염소' : 1, '뱀' : 1, '수영장' : 1,'테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,},
            '강아지' : {'고양이' : 1, '개' : 1, '귀엽다' : 1, '동물' : 1, '빗다' : 1, '거북이' : 1, '뱀' : 1, '토끼' : 1,'테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,},
            '고양이' : {'개' : 1, '귀엽다' : 1, '동물' : 1, '토끼' : 1, '쥐' : 1, '거북이' : 1, '빗다' : 1, '염소' : 1,'테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,},
            '거울' : {'빗다' : 1,'얼굴' : 1,'입다' : 1,'옷장' : 1,'세탁기' : 1,'꾹' : 1,'식탁' : 1,'예쁘다' : 1,'테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,},
            '칫솔' : {'비누' : 1,'부릉' : 1,'걸레' : 1,'닦다' : 1,'껄떡껄떡' : 1,'찰랑찰랑' : 1,'수건' : 1,'빗자루' : 1,'테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,},
            '양말' : {'모자' : 1,'신다' : 1,'옷' : 1,'장갑' : 1,'구두' : 1,'신발' : 1,'후드득' : 1,'바지' : 1,'테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,},
            '선물' : {'생일' : 1, '잠옷' : 1, '주다' : 1,'상자' : 1,'사다' : 1,'크리스마스' : 1,'옹기종기' : 1,'열쇠' : 1,'테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,},
            '꽃' : {'활짝' : 1,'나비' : 1,'나무' : 1,'봄' : 1,'예쁘다' : 1,'가을' : 1,'쫙' : 1,'목도리' : 1,'테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,},
            '텔레비전' : {'틀다' : 1,'보다' : 1,'라디오' : 1,'에어컨' : 1,'켜다' : 1,'맛보다' : 1,'핸드폰' : 1,'집' : 1,'테스트1':1, '테스트2':1, '테스트3':1, '테스트4':1,}
        }

        ## init recommendation (first send)
        self.get_recommendations()
        
        print('Init modelserver done')

 
    def get_attended_objects(self, focus, object_bboxes, classes):
        ''' Returns the object at the `focus` coordinate.

            Epsilon widens the bbox by a fixed amount to accomodate slight
            errors in the gaze coordinates.

        '''
        attended_objects = []
        x, y = focus
        epsilon = 10 # widens the bounding box, pixels

        for i, bbox in enumerate(object_bboxes):
            # exclude person!
            if classes[i] == 'person':
                continue

            left, top, right, bottom = bbox
            if (x > left-epsilon and x < right+epsilon and 
               y < bottom+epsilon and y > top-epsilon):
                attended_objects.append(classes[i])

        return attended_objects
        

    def update_context(self, modality, targets):
        ''' Adds weight to the detected objects: by `alpha` for a single visual
            frame and by `beta` for an audial utterance.
        '''
        if modality not in {'visual', 'audial'}:
            raise ValueError('modality must be one of visual or audial')
        

        ## get distribution
        target_length = 0

        target_dist = defaultdict(float)
        for t in targets:
            if t in self.objects:
                target_dist[t] += 1
                target_length += 1
        
        alpha = 0.017
        beta = 0.1

        if target_length != 0:
            for o in self.objects:
                if modality == 'visual':
                    self.context[o] = self.context[o] * (1-alpha) + target_dist[o] * alpha / target_length
                if modality == 'audial':
                    self.context[o] = self.context[o] * (1-beta) + target_dist[o] * beta / target_length
        
        recommendations = self.get_recommendations()

        return recommendations

    def get_recommendations(self):
        ''' Returns a list of recommended words.

            The proportion of words are determined by the context weights.

            A max heap-like structure would be a lot more convenient than
            recalculating weights and sorting every time...
        '''
        N = 12 # Total number of words to recommend
        N_h = N/2
        count = N_h
        
        recommendations = [] # list to order


        for item in sorted(self.context.items(), key = lambda x: x[1], reverse=True):
            obj = item[0]
            weight = item[1]
            
            # number of targets to recommend for this word
            n = math.ceil(weight * N_h)
            if count == 0:
                break
            elif count - n < 0:
                n = count
            count -= n
            
            heap_candidates = heapq.nlargest(int(n*2), self.candidates[obj].items(), key = lambda x : x[1])
            top_candidates = [ c[0] for c in heap_candidates]
            recommendations.append({'object' : obj, 'target_words' : top_candidates})
        
        recommendations = sorted(recommendations, key = lambda x: len(x['target_words']), reverse=True)
        
        recommendation_to_queue = {
            'tag' : 'recommendation',
            'recommendation' : recommendations
        }
        
        if self.result_queue:
            self.result_queue.put(recommendation_to_queue)
            
        return recommendations

    
    def on_spoken(self, words):
        ''' Action for when a target word is spoken. TODO

            * This isn't the name of the object! It's the candidate.

            The word's relevance should be decreased a bit so that the parent
            diversifies words.
        '''
        gamma = 0.01 # amount to decrement the relevance by 
        
        target_spoken = []

        for word in words:
            for obj in self.candidates:
                for cand in self.candidates[obj]:
                    if cand == word:
                        self.candidates[obj][cand] = round(self.candidates[obj][cand] - gamma,1)
                        target_spoken.append(word)
            
        if len(target_spoken) > 0:
            
            target_to_queue = {
                'tag':'target_words',
                'words':target_spoken
            }

            if self.result_queue:
                self.result_queue.put(target_to_queue)
        return target_spoken

    def run(self, visualize=False):
        ''' Main loop.
            Visualize only works in Jupyter.
        '''
        # These processes should be joined on error, interrupt, etc.
        self.visual_process_1.start() # start processes
        # self.visual_process_2.start() # start processes
        # self.visual_process_3.start() # start processes


        self.audial_process.start()
        print('process start')
        # This is unnecessary because the queue.get() below is blocking anyways
        # self.barrier.wait()

        transcript = ''
        spoken_words_prev = []
        spoken_words_update = []
        target_spoken = []

        recommendations = []

        image = None
        object_bboxes = []
        object_confidences = []
        object_classnames = []
        face_bboxes = []
        gaze_targets = []


        while (1):
            # This blocks until an item is available
            result = self.queue.get(block=True, timeout=None) 

            if result['from'] == 'image':                    
                image = result['image']
                object_bboxes = result['object_bboxes']
                object_confidences = result['object_confidences']
                object_classnames = result['object_classnames']
                face_bboxes = result['face_bboxes']
                gaze_targets = result['gaze_targets']
                camera_id = result['camera_id']
                frame_num = result['frame_num']


                attended_objects = [] # includes both parent & child for now
                for target in gaze_targets:
                    attended_objects.extend(self.get_attended_objects(
                            target, object_bboxes, object_classnames))
                
                target_objects = []
                for o in attended_objects:
                    if o in self.visual_classes.keys():
                        object_korean = self.visual_classes[o]
                        target_objects.append(object_korean)
                
                # update if there's objects
                if len(target_objects) != 0:
                    recommendations = self.update_context('visual', target_objects)

                if visualize and camera_id == '1':
                    self.visualizer.clear()
                    self.visualizer.draw_objects(image, object_bboxes, 
                            object_classnames, object_confidences)
                    self.visualizer.draw_face_bboxes(image, face_bboxes)
                    for i, face_bbox in enumerate(face_bboxes):
                        self.visualizer.draw_gaze(image, face_bbox, 
                                gaze_targets[i])

                    image = self.visualizer.add_captions_recommend(image, transcript, target_spoken, recommendations)
                    # self.visualizer.imshow(image)
                    self.visualizer.visave(image, frame_num)
                target_spoken.clear()

            elif result['from'] == 'audio':
                transcript = result['transcript']
                
                spoken_words = self.morph_analyze(transcript)
                
                spoken_words_update = spoken_words.copy()
                
                ## TODO : change to regex  
                for word in spoken_words_prev:
                    if word in spoken_words_update:
                        spoken_words_update.remove(word)
                
                # update spoken & target word weight
                spoken = self.on_spoken(spoken_words_update)
                if len(spoken) != 0:
                    target_spoken = spoken


                # if transcript is final
                if result['is_final']: 
                    spoken_objects = []

                    # update spoken objects list
                    for word in spoken_words:
                        if word in self.objects:
                            spoken_objects.append(word)

                    # update object context
                    if len(spoken_objects) != 0 :
                        recommendations = self.update_context('audial', spoken_objects)
                    
                    spoken_words.clear()
                        
                spoken_words_prev = spoken_words
                        
    def morph_analyze(self, transcript):
        spoken_words = []
        
        try:
            line_pos = self.Khaiii_api.analyze(transcript)

            for w in line_pos:
                for m in w.morphs:
                    if m.tag in ['NNG', 'NR', 'MAG']:
                        spoken_words.append(m.lex)
                    elif m.tag in ['VV', 'VA']:
                        spoken_words.append(m.lex + '다')
                    elif m.tag in ['XR']:
                        spoken_words.append(m.lex+'하다')
                    else :
                        spoken_words.append(m.lex)
        except:
            print('morph')
        
        return spoken_words


def start(url, queue, is_visualize):
    print('server start')
    server = ModelServer(url, queue)

    server.run(visualize=is_visualize)

async def pop_and_send(websocket, path):
    print("websocket start!")

    while(1):
        data = data_queue.get(block=True)
        data_json = json.dumps(data,ensure_ascii=False)

        await websocket.send(data_json)
        

if __name__ == '__main__': 
    
    server_process = Process(target=start, 
                args=(url, data_queue, True))
    
    server_process.start()

    while (1):
        result = data_queue.get(block=True)
        print(result)
        print('\n')
#     websocket_server = websockets.serve(pop_and_send, '0.0.0.0', 8888, ping_interval = None)

#     asyncio.get_event_loop().run_until_complete(websocket_server)
#     asyncio.get_event_loop().run_forever()
    
    
    
