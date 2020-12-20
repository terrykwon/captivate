from modelserver.base.base_predictor import BasePredictor

from visatt.model import ModelSpatial
from visatt.utils import imutils, evaluation
from visatt.config import *

from PIL import Image
import torch
from torchvision import datasets, transforms
import numpy as np
from skimage.transform import resize


def _get_transform():
    transform_list = []
    
    # input_resolution=224
    transform_list.append(transforms.Resize((input_resolution, 
                                             input_resolution))) 
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

test_transforms = _get_transform()


class GazeFollower(BasePredictor):

    def __init__(self, weight_path='models/weights/visatt.pt'):
        # Visual attention
        self.model = ModelSpatial()
        state_dict = self.model.state_dict()

        if torch.cuda.is_available():
            pretrained_state_dict = torch.load(weight_path)['model']
        else:
            pretrained_state_dict = torch.load(weight_path, 
                                    map_location=torch.device('cpu'))['model']

        state_dict.update(pretrained_state_dict)
        self.model.load_state_dict(state_dict)
        self.model.train(False)

        if torch.cuda.is_available():
            self.model.cuda()

    
    def predict_gaze(self, image, face_bbox):
        """ image must be PIL image :(
        """
        width, height = image.size

        head = image.crop(face_bbox)
        head = test_transforms(head)
        frame = test_transforms(image)

        head_channel = imutils.get_head_box_channel(face_bbox[0], face_bbox[1], 
                        face_bbox[2], face_bbox[3], width, height,
                        resolution=input_resolution).unsqueeze(0)

        if torch.cuda.is_available():
            head = head.unsqueeze(0).cuda()
            frame = frame.unsqueeze(0).cuda()
            head_channel = head_channel.unsqueeze(0).cuda()
        else:
            head = head.unsqueeze(0)
            frame = frame.unsqueeze(0)
            head_channel = head_channel.unsqueeze(0)

        # 3 inputs to model
        raw_hm, _, inout = self.model(frame, head_channel, head) 

        # heatmap modulation
        # detach from computational graph
        raw_hm = raw_hm.cpu().detach().numpy() * 255 
        raw_hm = raw_hm.squeeze()
        inout = inout.cpu().detach().numpy()
        inout = 1 / (1 + np.exp(-inout))
        inout = (1 - inout) * 255
        norm_map = resize(raw_hm, (height, width)) - inout

        point = evaluation.argmax_pts(norm_map) 

        return point

        # return norm_map, raw_hm, inout


    def strongest_pixel(self, width, height, raw_hm):
        """ Returns a single pixel where the focus is predicted to be strongest.
        """
        pred_x, pred_y = evaluation.argmax_pts(raw_hm)
        pred_x, pred_y = [pred_x/output_resolution * width, 
                          pred_y/output_resolution * height]

        return pred_x, pred_y