
import pandas as pd
import numpy as np
from collections import defaultdict

class Guidance : 

    def __init__(self, file_path):

        self.guide_dict = self.read_guide_csv(file_path)
        
        self.object_names = list(self.guide_dict.keys())


    def read_guide_csv(self, file_path):
        guide_dict = defaultdict(dict)

        guidance_arr = np.array(pd.read_csv(file_path))

        for guide in guidance_arr:
            object_name = guide[0]
            word = guide[1]
            sentence = guide[2]
            highlight = guide[3]
            id = guide[4]
            color = guide[5]

            guide_dict[object_name][word] = dict({'sentence': sentence, 'highlight': highlight, 'id': id, 'color' : color, 'weight': 1})


        return dict(guide_dict)

    def get_object_names(self):

        return list(self.guide_dict.keys())

    def get_object_context(self):
        object_list = self.get_object_names()
        
        object_context = {obj : 1/len(object_list) for obj in object_list}

        return object_context

    def get_candidates(self):
        
        return self.guide_dict

