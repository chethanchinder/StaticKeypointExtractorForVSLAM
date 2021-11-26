import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import  get_cfg
from detectron2.data import  MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import time

import cv2
import numpy as np

class Detectron:
	def __init__(self):
		self.cfg = get_cfg()
		self.cfg.MODEL.DEVICE='cpu'
		# load model config and pretrained model
		self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml"))
		self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml")
		self.predictor = DefaultPredictor(self.cfg)
	def onImage(self,image):
		panoptic_seg, segmentation_info = self.predictor(image)["panoptic_seg"]
		# v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
		# out = v.draw_panoptic_seg_predictions(panoptic_seg,segmentation_info)
		# cv2.imshow("output",out.get_image()[:, :, ::-1])
		# cv2.waitKey(1)
		return panoptic_seg
if __name__=="__main__":
	model_start = time.time()
	detectron_seg = Detectron()
	model_end = time.time()
	print(model_end-model_start)
	cap = cv2.VideoCapture("images/test_countryroad.mp4")
	while cap.isOpened():
		ret,image = cap.read()
		if ret:
			image = cv2.resize(image, (640,480))
			start = time.time()
			detectron_seg.onImage(image)
			end = time.time()
			print("each frame time: ", end-start)
