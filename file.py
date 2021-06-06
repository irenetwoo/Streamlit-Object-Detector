import cv2
import streamlit as st
import shutil
import os
import numpy as np
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
setup_logger()
from PIL import Image
import torch, torchvision

st.markdown("""<style> .font {font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;}</style>""", unsafe_allow_html=True)
st.markdown('<p class="font">Name The Objects</p>', unsafe_allow_html=True)
st.markdown('** Find the items in the picture labelled wrongly **')
st.markdown('1. Upload a photo')
st.markdown('2. A button will later appear at the bottom. Click it !')


@st.cache(suppress_st_warning=True, persist=True, max_entries=10, ttl=3600)
def initialization():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml")
    predictor = DefaultPredictor(cfg)
    return cfg, predictor

@st.cache(suppress_st_warning=True, persist=True, max_entries=10, ttl=3600)
def inference(predictor, img):
    return predictor(img)


@st.cache(suppress_st_warning=True, persist=True, max_entries=10, ttl=3600)
def output_image(cfg, img, outputs):
    metadata_ = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=3.0)
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    pred_classes = outputs['instances'].pred_classes.cpu().tolist()
    pred_class_names = list(map(lambda x: class_names[x], pred_classes))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_img = out.get_image()
    return processed_img

def main():
    cfg, predictor = initialization()
    uploaded_img = st.file_uploader("Upload a photo here", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(uploaded_img)
        st.image(image, caption=' Before', use_column_width=True)
        st.write("waiting for a button . . .")
        outputs = inference(predictor, img)
        out_image = output_image(cfg, img, outputs)
        if st.button("click me !"):
          st.markdown('<p class="font"> TADAA ! </p>', unsafe_allow_html=True)
          st.image(out_image, caption=' After ', use_column_width=True)   
          st.markdown(' **Can you find any item(s) labelled wrongly in the picture ?** ')
          st.markdown('')
          st.markdown('By [IreneToo](https://github.com/irenetwoo/Streamlit-Object-Detector)')
          st.write('This app uses a part of detection code from  Javier Esteve repo https://github.com/xavialex/Detectron2-Instance-Segmentation. Many thanks to this project.')
          
if __name__ == '__main__':
    main()

    



