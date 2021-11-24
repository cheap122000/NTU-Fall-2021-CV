import cv2
from PIL import Image
import random
import numpy as np

def GetGaussianNoise_Image(original_Image, amplitude):
    gaussianNoise_Image = original_Image.copy()
    for c in range(original_Image.size[0]):
        for r in range(original_Image.size[1]):
            noisePixel = int(original_Image.getpixel((c, r)) + amplitude * random.gauss(0, 1))
            noisePixel = 255 if noisePixel > 255 else noisePixel
            gaussianNoise_Image.putpizel((c, r), noisePixel)
    return gaussianNoise_Image

def GetSaultAndPepper_Image(original_Image, threshold):
    SaltAndPepper_Image = original_Image.copy()

    return SaltAndPepper_Image