import os
import json
import cv2
import numpy as np
from typing import List, Tuple

import torch
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from PIL import Image
