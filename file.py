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

