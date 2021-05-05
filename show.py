import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
import cv2
import copy
import random

parts_list = [[100, 39], [108, 39], [87,39], [120, 39], [77,39], [134,74], [60,74], 
[141,113], [57,113], [138, 142], [56,142], [117, 143], [82, 143], [114, 193], [83, 193], [116, 237], [82, 237]]
nose = [100, 39]
l_eye = [108, 39]
r_eye = [87,39]
l_ear = [120, 39]
r_ear = [77,39]
l_shoulder = [134,74]
r_shoulder = [60,74]
l_elbow = [141,113]
r_elbow = [57,113]
l_hand = [138, 142]
r_hand = [56,142]
l_waist = [117, 143]
r_waist = [82, 143]
l_knee = [114, 193]
r_knee = [83, 193]
l_ankle = [116, 237]
r_ankle = [82, 237]
print(parts_list)
# image_path = "/home/ono/Downloads/istockphoto-823826708-1024x1024.jpg"
# image = cv2.imread(image_path)

# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()