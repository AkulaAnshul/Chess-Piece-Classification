import cv2
from skimage.feature import hog
import numpy as np

def extract_hog_features(images):
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feat = hog(
            gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
            visualize=False, multichannel=False
        )
        features.append(feat)
    return np.array(features)
