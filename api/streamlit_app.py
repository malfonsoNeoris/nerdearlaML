from functools import partial
from detectors.yolo_detector import YoloDetector
from detectors.fastai_classifier import FastAiClassifier


import cv2
import numpy as np
from fastai.vision.all import PILImage
from PIL import Image as PImage
from itertools import groupby
import math

import streamlit as st


DETECTOR='DETECTOR'
CLASSIFIER='CLASSIFIER'

SIDEBAR_OPTIONS = [DETECTOR,CLASSIFIER]

class Detector:
    def __init__(self, draw=True):
        weights_file = "models/detector/y4t-custom.weights"
        names_file = "models/detector/coco.names"
        cfg_file = "models/detector/y4t-custom.cfg"

        self.detector = YoloDetector(names_file, weights_file,cfg_file, input_size = (416, 416) )
        self.img = self.get_image_from_upload()
        if self.img is not None:
            with st.sidebar:
                thresh = st.slider('thresh', .2, 1.0, .7, step=.05)
                nms_tresh=st.slider('nms_thresh', .2, 1.0, .4, step=.1)
                draw = st.checkbox('draw?', value=True)
                predictions = self.get_prediction(thresh, nms_tresh,draw)
                st.write(predictions)
            self.display_output()

    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')

    def get_prediction(self,conf, nms_tresh, draw):
        open_cv_image = np.array(self.img).copy() 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        if st.button('Detect'):
            detections = self.detector.DetectOnImage(open_cv_image, conf, nms_tresh)
            if draw and detections is not None:
                self.detector.DrawDetections(open_cv_image, detections)
            open_cv_image = open_cv_image[:, :, ::-1]
            self.img = PImage.fromarray(open_cv_image)
            return detections
        else: 
            st.write(f'Click the button to detect') 
            return None


class Classifier:
    def __init__(self):
        filename = 'models/classifier/classifier.pkl'
        self.classifier = FastAiClassifier(filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            with st.sidebar:
                self.get_prediction()
            self.display_output()
            
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')

    def get_prediction(self):

        if st.button('Classify'):
            data = self.classifier.predict(self.img)
            st.write(data)
        else: 
            st.write(f'Click the button to classify') 


if __name__=='__main__':


    st.title("Model visualizer demo")
    
    with st.sidebar:
        st.header("Configuration")
        mode = st.selectbox("Select Model", SIDEBAR_OPTIONS)
    if mode == DETECTOR:
        predictor = Detector()
    elif mode == CLASSIFIER:
        predictor = Classifier()
    else:
        st.warning("Invalid Option")



