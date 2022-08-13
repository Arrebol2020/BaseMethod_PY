from superpoint_method import superpoint_m
from keynet_method import keynet_m
from d2net_method import d2net_m
from r2d2_method import r2d2_m
from sift_method import sift_m


if __name__ == "__main__":
  img_path = r"E:\datasets\bop_datasets\ycbv\crop_test\21\img\000057_000001.png"

  try:
    superpoint_m(img_path)
  except:
    print("superpoint_m failed")
  
  try:
    keynet_m(img_path)
  except:
    print("keynet_m failed")

  try:
    r2d2_m(img_path)
  except:
    print("r2d2_m failed")

  try:
    d2net_m(img_path)
  except:
    print("d2net_m failed") 
  
  try:
    sift_m(img_path)
  except:
    print("d2net_m failed") 
  
  
