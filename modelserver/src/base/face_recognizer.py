from arcface.models import *
from arcface.config import Config

from torch.nn import DataParallel


class FaceRecognizer(): 

    def __init__(self, weight_path, target_faces):
        config = Config()
        self.model = resnet_face18(config.use_se)
        state_dict = torch.load(weight_path)
        self.model.load_state_dict(state_dict)

        self.model.eval()

    def predict(self, candidate_bboxes):
        pass