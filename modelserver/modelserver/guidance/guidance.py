
import pandas as pd
import numpy as np
from collections import defaultdict

import time

toy_num = 12

class Toy:
    def __init__(self, toy_name):
        self.toy_name = toy_name
        self.weight = 1/toy_num

        self.phrases = []

    def add_phrase(self, phrase):
        self.phrases.append(phrase)
    
    def update_weight(self, target_dist, target_length, alpha):
        
        self.weight = self.weight * (1-alpha) + target_dist[self.toy_name] * alpha / target_length

    def is_phrase_spoken(self, word, display_phrases):

        for phrase in self.phrases:
            if (phrase.phrase in display_phrases) and (phrase.word == word):
                phrase.increase_spoken_count()
                return True

        return False

    def track_displayed_time(self, curr_time):
        need_ordered = False

        for phrase in self.phrases:
            if (phrase.is_displayed == True) and (phrase.track_displayed_time(curr_time)):
                need_ordered = True
        return need_ordered
    
    def set_display(self, displayed_phrases):
        
        for phrase in self.phrases:
            if phrase.phrase in displayed_phrases:
                if phrase.is_displayed == True:
                    pass
                elif phrase.is_displayed == False:
                    phrase.is_displayed = True
                    phrase.on_displayed()
                else:
                    print("is_displayed error")
                
            else:
                phrase.is_displayed = False
            





class Phrase:
    def __init__(self, word, phrase, id, highlight, color):
        self.word = word
        self.phrase = phrase
        self.highlight = highlight
        self.id = id
        self.weight = 1
        self.color= color
        
        self.start_displayed = 0
        self.spoken_count = 0

        self.is_displayed = False

    
    def increase_spoken_count(self):
        self.spoken_count += 1
        self.weight -= 0.5

        if self.spoken_count > 1:
            self.is_displayed = False
            self.spoken_count = 0 
    
    def on_displayed(self):
        self.start_displayed = int(round(time.time() * 1000))

    def track_displayed_time(self, curr_time):
        if curr_time - self.start_displayed > 120000:
            self.weight -= 1
            return True
        return False
    
    def print_all(self):
        print("-----------------------------")
        print("phrase : "+self.phrase)
        print("weight : "+str(self.weight))
        print("start_displayed : "+ str(self.start_displayed))
        print("spoken_count : "+str(self.spoken_count))

class Guidance : 

    def __init__(self, file_path):

        self.toys = []
        self.toy_list = []

        self.read_guide_csv(file_path)


    def read_guide_csv(self, file_path):
        
        guidance_arr = np.array(pd.read_csv(file_path))

        for guide in guidance_arr:
            toy_name = guide[0]

            if toy_name not in self.toy_list:
                self.toy_list.append(toy_name)

                new_toy = Toy(toy_name)
                self.toys.append(new_toy)

            word = guide[1]
            sentence = guide[2]
            highlight = guide[3]
            id = guide[4]
            color = guide[5]

            new_phrase = Phrase(word, sentence, id, highlight, color)

            for toy in self.toys:
                if toy.toy_name == toy_name:
                    toy.add_phrase(new_phrase)

    def get_toys(self):
        return self.toys

    # def get_object_names(self):

    #     return list(self.guide_dict.keys())

    # def get_object_context(self):
    #     object_list = self.get_object_names()
        
    #     object_context = {obj : 1/len(object_list) for obj in object_list}

    #     return object_context

    # def get_candidates(self):
        
    #     return self.guide_dict

