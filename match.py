import cv2
import numpy as np
from numpy import linalg as LA
import pysift

def get_matches(base_img, sec_img):

    base_img_kpts, base_img_desc = pysift.computeKeypointsAndDescriptors(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY))
    sec_img_kpts, sec_img_desc = pysift.computeKeypointsAndDescriptors(cv2.cvtColor(sec_img, cv2.COLOR_BGR2GRAY))
    # Sift = cv2.SIFT_create()
    # base_img_kpts, base_img_desc = Sift.detectAndCompute(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), None)
    # sec_img_kpts, sec_img_desc = Sift.detectAndCompute(cv2.cvtColor(sec_img, cv2.COLOR_BGR2GRAY), None)
    good_matches=[]
    for i, feat in enumerate(base_img_desc):
        distances = LA.norm(sec_img_desc-feat, axis=1)
        nn = np.argsort(distances)[:2]
        dist1, dist2 = distances[nn[0]], distances[nn[1]]
        if dist1/max(1e-6, dist2) < 0.75:
            good_matches.append([i,nn[0]])

    return good_matches, base_img_kpts, sec_img_kpts