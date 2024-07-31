import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import nrrd
import nibabel as nib
import skimage.measure
import trimesh
from scipy.ndimage import zoom
import RRTS as turtle_hiding
import torch
import torch.nn as nn
import torchvision
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
import IFRL as ImprovedCompressed
import nrrd
import nibabel as nib
import skimage.measure
import trimesh
from scipy.ndimage import zoom
import utils_RRTS as turtle_tools
from tqdm import tqdm
import pandas as pd

# 生成秘密数据
def generate_binsecret(secret_length=3):
    bin_secret=[]
    for _ in range(secret_length):
        bin_secret.append(np.random.randint(0,2))
    return bin_secret


# hiding secret images
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

module_conceal = SamStegNet.Module_conceal().cuda()
module_reveal = SamStegNet.Module_reveal().cuda()

module_conceal.load_state_dict(torch.load("Module_conceal.pth"), strict=False)
module_reveal.load_state_dict(torch.load("Module_reveal.pth"), strict=False)

