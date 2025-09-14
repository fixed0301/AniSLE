import os
import json
import cv2
import numpy as np
from typing import List, Tuple

# SAM (segment-anything)
# from segment_anything import SamPredictor, sam_model_registry

# Diffusers / Hugging Face
import torch
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from PIL import Image
