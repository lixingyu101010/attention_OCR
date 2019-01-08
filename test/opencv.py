import cv2
import sys
import os
from PIL import Image
sys.path.append(os.getcwd())

image = cv2.imread('test/test_image/1.png')
image = cv2.copyMakeBorder(image, 0, 0, 0, 400-image.shape[1], cv2.BORDER_CONSTANT, value=[255, 255, 255])
img = Image.fromarray(image)
img.show()