""" functions to draw plots
"""

from visatt.utils import evaluation
from visatt.config import *

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import IPython
from io import StringIO, BytesIO
import PIL
from PIL import ImageFont, ImageDraw
import textwrap


class Visualizer:

    @classmethod
    def hex2color(h):
        return 


    COLORS_HEX = { # class static variables are declared outside any method
        'white': '#ffffff',
        'lime': '#A0ED2F',
        'cyan': '#00ffff',
        'blue': '#0398fc',
        'pink': '#ff619e',
        'orange': '#ff9812',
        'red': '#ff4545',
        'green': '#00ff8c',
    }

    # converts hex to rgb
    COLORS_RGB = {k: tuple(int(v[i:i+2], 16) for i in (1, 3, 5))\
                  for (k,v) in COLORS_HEX.items()}


    def __init__(self, scale=1.0):
        ''' Font sizes and line widths are normalized, i.e. resized in
            proportion to the image dimensions.

            They can be subsequently scaled using `scale`.
        '''
        self.scale = scale
        self.font = ImageFont.truetype('/workspace/modelserver/modelserver/visualization/NotoSansCJKkr-Regular.otf', 32)


    def clear(self):
        IPython.display.clear_output(wait=True)


    def imshow(self, image, width=640):
        ''' Shows image in jupyter, resizing to fit `width`
        '''
        f = BytesIO()

        scale = width / image.shape[1] # (height, width, channels)! 

        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        resized = cv2.resize(image, (width, height))
        PIL.Image.fromarray(resized).save(f, 'jpeg')
        IPython.display.display(IPython.display.Image(data=f.getvalue()))

        


    def add_captions(self, image, transcript, attended_objects, target_spoken):
        width = image.shape[1]
        height = image.shape[0]
        scale = width / 1280 * self.scale
        dy = 50
        dx = 100

        box = np.zeros((height//2, width, 3)).astype('uint8')
        box = self.write_text(box, transcript, (20, 10), 
                              Visualizer.COLORS_RGB['white'])

        objects = ', '.join(attended_objects)
        context_string = 'Context: {}'.format(objects)

        box = self.write_text(box, context_string, (20, 110), 
                              Visualizer.COLORS_RGB['pink'])

        targets = ', '.join(target_spoken)
        target_string = 'Target word: {}'.format(targets)

        box = self.write_text(box, target_string, (20, 210), 
                              Visualizer.COLORS_RGB['pink'])

        stacked = np.vstack((image, box))

        return stacked
    
    def add_captions_recommend(self, image, transcript, target_spoken, recommendations):
        width = image.shape[1]
        height = image.shape[0]
        scale = width / 1280 * self.scale
        dy = 50
        dx = 100

        box = np.zeros((height//2, width, 3)).astype('uint8')
        box = self.write_text(box, transcript, (20, 10), 
                              Visualizer.COLORS_RGB['white'])

        targets = ', '.join(target_spoken)
        target_string = 'Target word: {}'.format(targets)

        box = self.write_text(box, target_string, (20, 110), 
                              Visualizer.COLORS_RGB['white'])
        
        rec_string = ''
        for k, v in recommendations:
            rec_string += "{} : {} / ".format(k, ', '.join(v))
        
        box = self.write_text(box, rec_string, (20, 210), 
                              Visualizer.COLORS_RGB['white'])
        
        stacked = np.vstack((image, box))

        return stacked
    


    def draw_objects(self, image, bboxes, classes, scores):
        ''' If bboxes is empty, returns the original image.
        '''
        width = image.shape[1]
        height = image.shape[0]
        scale = width / 1280 * self.scale

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1.0 * scale
        font_thickness = int(2 * scale)
        font_color = Visualizer.COLORS_RGB['blue']

        box_thickness = int(2 * scale)
        box_color = Visualizer.COLORS_RGB['cyan']

        if bboxes == []:
            return
        
        for i, bbox in enumerate(bboxes):
            classname = classes[i]

            # person boxes are annoying, so exclude them for now

            if classname == 'person': 
                continue

            left, top, right, bottom = np.around(bbox).astype('int')

            image = cv2.rectangle(image, (left, top), (right, bottom), 
                    box_color, box_thickness)

            score = scores[i]
            text_bottom_left = (left, top)
            text = '{} {:.2f}'.format(classname, score)
            image = cv2.putText(image, text, text_bottom_left, font, fontScale,  
                            font_color, font_thickness, cv2.LINE_AA, False)


    def draw_gaze(self, image, face_bbox, focus):
        """ face_bbox: [float, float, float, float]
        """ 
        width = image.shape[1]
        height = image.shape[0]
        scale = width / 1280 * self.scale

        color = Visualizer.COLORS_RGB['red']
        thickness = int(2 * scale) # in px 
        radius = int(12 * scale)

        left, top, right, bottom = np.around(face_bbox).astype('int')
        focus = tuple(np.around(focus).astype('int'))

        # start point is just middle of face bbox
        start_point = ((left+right)//2, (bottom+top)//2)
        
        cv2.line(image, start_point, focus, color, thickness)
        cv2.circle(image, focus, radius, color, thickness)


    def write_text(self, image, text, location, color):
        ''' location: bottom left of text
        '''
        # font = cv2.FONT_HERSHEY_SIMPLEX
        (x, y) = location
        dy = 50

        image_pil = PIL.Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)
        wrapped = textwrap.wrap(text, width=40)

        for line in wrapped:
            y += dy
            draw.text((x, y), line, font=self.font, fill=color)
        
        # in-place! image doesn't actually need to be returned
        # image = cv2.putText(image, text, location, font, fontScale,  
        #                 color, thickness, cv2.LINE_AA, False)
        image = np.array(image_pil)

        return image


    def draw_face_bboxes(self, image, bboxes):
        ''' If bboxes is empty, returns the original image.
        '''
        width = image.shape[1]
        height = image.shape[0]
        scale = width / 1280 * self.scale
        color = Visualizer.COLORS_RGB['green']
        thickness = int(2 * scale) # in px 

        if bboxes == []:
            return image
        
        for bbox in bboxes:
            left, top, right, bottom = np.around(bbox).astype('int')
            image = cv2.rectangle(image, (left, top), (right, bottom), 
                                  color, thickness)
        
        return image


    def plot_face_bboxes(self, image, face_bboxes, plot_image=True):
        if face_bboxes.shape == (4,):
            face_bboxes = np.expand_dims(face_bboxes, 0)

        fig = None
        if plot_image:
            fig, ax = plt.subplots(1, 1)
            plt.imshow(image)

        for face_bbox in face_bboxes:
            left, top, right, bottom = face_bbox
            ax = plt.gca()
            rect = patches.Rectangle((left,top), right-left, bottom-top, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

        return fig


    def plot_heatmap(self, image, norm_map, raw_hm, face_bbox, 
                     inout, plot_image=True):
        out_threshold = 100 # tunable parameter
        
        height, width, _ = image.shape
        
        if plot_image:
            fig, ax = plt.subplots(1,1)
            plt.imshow(image)
        else:
            ax = plt.gca()

        plt.imshow(norm_map, cmap='jet', alpha=0.2, vmin=0, vmax=255)

        if inout < out_threshold: # needs to be smaller than threshold
            pred_x, pred_y = evaluation.argmax_pts(raw_hm)
            norm_p = [pred_x/output_resolution, pred_y/output_resolution]
            circ = patches.Circle((norm_p[0]*width, norm_p[1]*height), 
                    height/50.0, facecolor=(0,1,0), edgecolor='none')
            ax.add_patch(circ)
            plt.plot((norm_p[0]*width,(face_bbox[0]+face_bbox[2])/2), 
                    (norm_p[1]*height,(face_bbox[1]+face_bbox[3])/2), 
                    '-', color=(0,1,0,1))


    def plot_object_segmentation(self, image, outputs, predictor):
        fig = plt.figure(figsize=(8,6))
        v = Visualizer(image, predictor.metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(out.get_image())