import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

path = '/home/aditya/hand_object_detector/images/test4.jpg'
path_output = '/home/aditya/hand_object_detector/images/test.jpg'
image = cv2.imread(path)
image = image[667:1853,829:2138,:]
bbox = np.load('bbox.npy')
cv2.imwrite('test.jpg',image)
