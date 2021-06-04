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
#assert torch.__version__.startswith("1.8") 

st.write('hello')
