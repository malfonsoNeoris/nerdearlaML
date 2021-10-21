# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

from detectron2.utils.visualizer import ColorMode
import glob
import random
import os
import cv2
import time

import numpy as np
from skimage import measure

from torch import topk


class BBox():
  def __init__(self, box, score, clas, cnt) -> None:
    self.left = box[0]
    self.top = box[1]
    self.bottom = box[3]
    self.right = box[2]
    self.score = score
    self.clas = clas
    self.cnt=cnt
  
  def box(self):
    return [self.left, self.top, self.right, self.bottom]
  
  def __str__(self) -> str:
    return f"rect:[{self.left},{self.top},{self.right},{self.bottom}] - score:{self.score:.2f} - class: {self.clas}"

  def __repr__(self) -> str:
    return str(self)

class Detectron2Detector:
  def __init__(self,path, classes, thres =0.8, device = 'cpu', type = 'COCO-Detection',network = 'faster_rcnn_X_101_32x8d_FPN_3x' ):
    self.classes = classes
    self._classes =  ["none"]+classes
    self.type = type
    self.network = network
    self.LoadModel(path, thres,device, type=type, network=network)

  def LoadModel(self, path, thres =0.8, device = 'cpu', type = 'COCO-Detection',network = 'faster_rcnn_R_50_FPN_3x'):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"{type}/{network}.yaml"))
    cfg.MODEL.WEIGHTS = path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thres   # set the testing threshold for this model
    cfg.MODEL.DEVICE=device
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self._classes)#+1 #no se porque me agrego una que es table..
    self.predictor = DefaultPredictor(cfg)
    MetadataCatalog.get(path).thing_classes = self._classes
    self.metadata=MetadataCatalog.get(path)

  @classmethod
  def close_contour(cls, contour):
    if not np.array_equal(contour[0], contour[-1]):
      contour = np.vstack((contour, contour[0]))
    return contour

  @classmethod
  def binary_mask_to_polygon(cls, binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
    binary_mask: a 2D binary numpy array where '1's represent the object
    tolerance: Maximum distance from original points of polygon to approximated
    polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """

    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = cls.close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

  def Predict(self, im, tresh=.8):
    self.predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=tresh

    return self.predictor(im)

  def getBoxes(self, outputs,  thres=0.8):

    instances = outputs['instances'].to('cpu')
    cont=0
    boxes=[]
    for i in range(len(instances)):
      if instances[i].scores<thres:
        continue
      #----------------
      cnt=None
      if len(instances[i].pred_masks)>0:
        mask = instances[i].pred_masks[0]
        # pad mask to close contours of shapes which start and end at an edge
        padded_binary_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
        cnts = measure.find_contours(padded_binary_mask, 0.5)
        cnts = np.subtract(cnts, 1)

        if len(cnts)>0:
          cnt=cnts[0]
          cnt = np.array(np.flip(cnt, axis=1)).astype(np.int32)


      #----------------

      box = instances[i].pred_boxes[0]
      box = box.tensor.flatten().tolist()
      box = [int(i) for i in box]
      score = float(instances[i].scores)
      clas  = int(instances[i].pred_classes)
      bbox = BBox(box, score, clas, cnt)
      boxes.append(bbox)


    return boxes

  def Draw(self,im,outputs, _class="all"):

    if(_class!="all"):
      index = self._classes.index(_class)
    else:
      index=-1
    
    v = Visualizer(im,
                      scale=1,
                      metadata = self.metadata
                      )
    
    instances = outputs['instances'].to('cpu')

    out = v.draw_instance_predictions(instances)
    return out.get_image()
    #return out
