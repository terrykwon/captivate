from modelserver.base.base_predictor import BasePredictor

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog





class ObjectDetector(BasePredictor):

    def __init__(self):
        # Object segmentation
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cuda"
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) #Get the basic model configuration from the model zoo 
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.MODEL.WEIGHTS = '/workspace/modelserver/models/train/weights/model_0002399.pth'
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12
        self.model = DefaultPredictor(cfg)


    def predict(self, image):
        outputs = self.model(image)
        return outputs


    def id_to_classname_mappings(self):
        # return self.model.metadata.thing_classes
        return ['dog','cat','fish','bear','flower','spoon','bicycle','shoe','ball','bus','bag','baby']
        # return ['ball','dog','cat','shoe','spoon','bowl','fork','bus','tv','bicycle','fish','mirror','toothbrush','sock','gift','flower']
