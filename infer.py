import numpy as np
import torch
import os
import json
import cv2
import glob
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from config import config


def infer_imgs(dataset, root, detection_root, model_path):
	cfg = config[dataset]
	dataset_root = os.path.join(root, dataset)
	detection_folder = os.path.join(detection_root, dataset)
	data_folder = os.path.join(dataset_root, cfg.test_folder)
	img_folder = cfg.img_folder
	img_ext = cfg.img_ext
	depth_ext = cfg.depth_ext

	scene_ids = sorted([int(scene_dir.name) for scene_dir in Path(data_folder).glob('*') if scene_dir.is_dir()])
	scene_cameras = defaultdict(lambda *_: [])

	for scene_id in tqdm(scene_ids, 'loading crop info'):
		scene_folder = os.path.join(data_folder, f'{scene_id:06d}')
		with open(os.path.join(scene_folder, 'scene_camera.json'), 'r') as json_file:
			scene_cameras[scene_id] = json.load(json_file)
	bboxes = np.loadtxt(os.path.join(detection_folder, 'bboxes.txt'), dtype='float32')
	obj_ids = np.loadtxt(os.path.join(detection_folder, 'obj_ids.txt'), dtype='int64')
	scene_ids = np.loadtxt(os.path.join(detection_folder, 'scene_ids.txt'), dtype='int64')
	view_ids = np.loadtxt(os.path.join(detection_folder, 'view_ids.txt'), dtype='int64')
	obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}

	xys_all = []
	q_features_all = []
	K_all = []
	obj_id_all =[]
	img_id_all = []
	scene_id_all = []

	sift = cv2.SIFT_create()
	for i in tqdm(range(len(bboxes)), 'infering'):
		scene_id, view_id, obj_id = scene_ids[i], view_ids[i], obj_ids[i]
		instance = dict(
			scene_id=scene_id, img_id=view_id, obj_id=obj_id, obj_idx=obj_idxs[obj_id],
			K=np.array(scene_cameras[scene_id][str(view_id)]['cam_K']).reshape((3, 3)),
			mask_visib=bboxes[i], bbox=bboxes[i].round().astype(int),
		)
		img_id = instance['img_id']
		fp = os.path.join(data_folder, f'{scene_id:06d}/{img_folder}/{img_id:06d}.{img_ext}')
		rgb = cv2.imread(str(fp), cv2.IMREAD_COLOR)[..., ::-1]  # BGR 2 RGB
		assert rgb is not None
		instance['rgb'] = rgb.copy()
		keypoints, q_features = sift.detectAndCompute(instance['rgb'], None)
		xys = [list(i.pt) for i in keypoints]

		xys_all.append(xys)
		q_features_all.append(q_features.tolist())
		K_all.append(instance["K"].tolist())
		obj_id_all.append(obj_id)
		if i == 3:
			break;
		
	return {"keypoints": xys_all, "features": q_features_all, "K_all": K_all, "obj_id_all": obj_id_all}



if __name__ == "__main__":
  #device = torch.device("cuda") if torch.is_available() else torch.device("cpu")
  model_path = ""
  dataset = "ycbv"
  root = r"C:\Users\DELL\Desktop\projs\BaseMethod_PY\data\bop"
  detection_folder = "C:\\Users\\DELL\\Desktop\\data\\detection_results_txt"
  infer_imgs(dataset, root, detection_folder, model_path)



