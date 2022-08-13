import cv2
import numpy as np

def sift_m(img_path):
  sift = cv2.SIFT_create()
  #img_path = r"E:\datasets\bop_datasets\ycbv\crop_test\15\img\000054_000094.png"
  img = cv2.imread(img_path, 0)

  keypoints, q_features = sift.detectAndCompute(img, None)
  img_keypoints = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
  rgb_img = cv2.imread(img_path)
  cv2.drawKeypoints(rgb_img, keypoints, img_keypoints)
  #cv2.imwrite("keynet.png", img_keypoints)
  cv2.imshow("sift", img_keypoints)
  cv2.waitKey(0)