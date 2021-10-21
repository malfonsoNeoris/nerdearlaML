from functools import partial
from yolo_detector import YoloDetector
from fastai_classifier import FastAiClassifier
from platemaskdetector import PlateMaskDetector
from plate_recognizer import PlateRecognizer

import cv2
import numpy as np
from fastai.vision.all import PILImage
from PIL import Image as PImage
from itertools import groupby
import math

import streamlit as st


VEHICLE_DETECTOR='VEHICLE DETECTOR'
VEHICLE_CLASSIFIER='VEHICLE CLASSIFIER'
VEHICLE_PROPERTIES_DETECTOR='PROPS DETECTION'
PLATE_WARPER = 'PLATE WARPER'
PLATE_RECOGNITION = 'PLATE RECOGNITION'
SIDEBAR_OPTIONS = [VEHICLE_DETECTOR,VEHICLE_CLASSIFIER, VEHICLE_PROPERTIES_DETECTOR, PLATE_WARPER, PLATE_RECOGNITION]

class VehicleDetector:
    def __init__(self, draw=True):
        weights_file = "models/vehicle_detection/yolov4-tiny-416.weights"
        names_file = "models/vehicle_detection/coco.names"
        cfg_file = "models/vehicle_detection/yolov4-tiny-416.cfg"

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


class TruckClassifier:
    def __init__(self):
        filename = 'models/truck_classifier/trucks.pkl'
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

class TruckPropertyrecognition:
    def __init__(self, draw=True):
        weights_file = "models/plate_and_others/yolo4-second.weights"
        names_file = "models/plate_and_others/obj.names"
        cfg_file = "models/plate_and_others/yolo4-second.cfg"

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

class PlateTransformer:
    def __init__(self, draw=True):
        file = 'models/plate_mask/patentes_mask.pth'

        self.detector = PlateMaskDetector(path =file )
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
        if hasattr(self, 'warped') and self.warped is not None:
            st.image(self.warped.to_thumb(320,160), caption='Aligned')  

    def get_prediction(self,conf, nms_tresh, draw):
        open_cv_image = np.array(self.img).copy() 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        if st.button('Detect'):
            outputs = self.detector.Predict(open_cv_image, conf)
            boxes = self.detector.getBoxes(outputs)
            if len(boxes)== 0:
                return  boxes
            box = boxes[0]
            warped = None
            pts=None
            if box.cnt is not None:
              warped, pts = self.detector.getWarped(open_cv_image,box )

            if draw: 
                cv2.circle(open_cv_image, tuple(pts[0].astype(int)), 3, (0, 0, 255), -1)
                cv2.circle(open_cv_image, tuple(pts[1].astype(int)), 3, (0, 255, 0), -1)
                cv2.circle(open_cv_image, tuple(pts[2].astype(int)), 3, (255, 0, 0), -1)
                cv2.circle(open_cv_image, tuple(pts[3].astype(int)), 3, (255, 255, 0), -1)
            #     cv2.drawContours(open_cv_image, [ctr], 0, (0,255,0), 2)

            if draw:
                img = self.detector.Draw(open_cv_image, outputs)
            
            st.write(boxes)
            open_cv_image = open_cv_image[:, :, ::-1]
            self.img = PImage.fromarray(open_cv_image)
            if warped is not None:
                warped = warped[:, :, ::-1]
                self.warped=PImage.fromarray(warped)
            
            return boxes
        else: 
            st.write(f'Click the button to detect') 

class PlateRecognition:
    def __init__(self, draw=True):
        weights_file = "models/plate_recognition/y4t-letters_final.weights"
        names_file = "models/plate_recognition/obj.names"
        cfg_file = "models/plate_recognition/y4t-letters.cfg"

        self.detector = PlateRecognizer(names_file, weights_file,cfg_file, input_size = (320, 160) )
        self.img = self.get_image_from_upload()
        if self.img is not None:
            with st.sidebar:
                thresh = st.slider('thresh', .1, 1.0, .2, step=.05)
                nms_tresh=st.slider('nms_thresh', .2, 1.0, .6, step=.1)
                draw = st.checkbox('draw?', value=True)
                predictions = self.get_prediction(thresh, nms_tresh,draw)
                if predictions is not None:
                    st.write(predictions)

                    plates = self.detector.getPlatesFromDetections(predictions)
                    st.write(plates)
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


if __name__=='__main__':


    st.title("Model visualizer demo")
    
    with st.sidebar:
        st.header("Configuration")
        mode = st.selectbox("Select Model", SIDEBAR_OPTIONS)
    if mode == VEHICLE_DETECTOR:
        predictor = VehicleDetector()
    elif mode == VEHICLE_CLASSIFIER:
        predictor = TruckClassifier()
    elif mode == VEHICLE_PROPERTIES_DETECTOR:
        predictor = TruckPropertyrecognition()
    elif mode == PLATE_WARPER:
        predictor = PlateTransformer()
    elif mode == PLATE_RECOGNITION:
        predictor = PlateRecognition()
    else:
        st.warning("Invalid Option")



