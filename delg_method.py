import cv2
import torch
import numpy as np
import pickle
import os

import sys
sys.path.append(r"C:\Users\DELL\Desktop\projs\BaseMethod_PY\method\delg")
from config import cfg
import config
import delg_utils


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    try:
        state_dict = checkpoint["model_state"]
    except KeyError:
        state_dict = checkpoint
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model
    model_dict = ms.state_dict()

    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    if len(pretrained_dict) == len(state_dict):
        print('All params loaded')
    else:
        print('construct model total {} keys and pretrin model total {} keys.'.format(len(model_dict), len(state_dict)))
        print('{} pretrain keys load successfully.'.format(len(pretrained_dict)))
        not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
        print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
    model_dict.update(pretrained_dict)
    ms.load_state_dict(model_dict)
    #ms.load_state_dict(checkpoint["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    #return checkpoint["epoch"]
    return checkpoint


def preprocess(im, scale_factor):
    im = im_scale(im, scale_factor) 
    im = im.transpose([2, 0, 1])
    im = im / 255.0
    im = color_norm(im, _MEAN, _SD)
    return im

def im_scale(im, scale_factor):
    h, w = im.shape[:2]
    h_new = int(round(h * scale_factor))
    w_new = int(round(w * scale_factor))
    im = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    return im.astype(np.float32)

def color_norm(im, mean, std):
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def setup_model():
    model = delg_utils.DelgExtraction()
    print(model)
    load_checkpoint(MODEL_WEIGHTS, model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def extract(im_array, model):
    input_data = torch.from_numpy(im_array)
    if torch.cuda.is_available(): 
        input_data = input_data.cuda() 
    fea = model(input_data, targets=None)
    _, delg_features, delg_scores = model(input_data, targets=None)
    #print(delg_features.size(), delg_scores.size())
    return delg_features, delg_scores


def delg_extract(img, model):
    """ multiscale process """
    # extract features for each scale, and concat.
    output_boxes = []
    output_features = []
    output_scores = []
    output_scales = []
    output_original_scale_attn = None
    for scale_factor in SCALE_LIST:
        im = preprocess(img.copy(), scale_factor)
        im_array = np.asarray([im], dtype=np.float32)
        delg_features, delg_scores = extract(im_array, model)
       
        #tmp = delg_scores.squeeze().view(-1)
        #print(torch.median(tmp))

        selected_boxes, selected_features, \
        selected_scales, selected_scores, \
        selected_original_scale_attn = \
                    delg_utils.GetDelgFeature(delg_features, 
                                        delg_scores,
                                        scale_factor,
                                        RF,
                                        STRIDE,
                                        PADDING,
                                        ATTN_THRES)

        output_boxes.append(selected_boxes) if selected_boxes is not None else output_boxes
        output_features.append(selected_features) if selected_features is not None else output_features
        output_scales.append(selected_scales) if selected_scales is not None else output_scales
        output_scores.append(selected_scores) if selected_scores is not None else output_scores
        if selected_original_scale_attn is not None:
            output_original_scale_attn = selected_original_scale_attn
    if output_original_scale_attn is None:
        output_original_scale_attn = im.clone().uniform()
    # concat tensors precessed from different scales.
    output_boxes = delg_utils.concat_tensors_in_list(output_boxes, dim=0)
    output_features = delg_utils.concat_tensors_in_list(output_features, dim=0)
    output_scales = delg_utils.concat_tensors_in_list(output_scales, dim=0)
    output_scores = delg_utils.concat_tensors_in_list(output_scores, dim=0)
    # perform Non Max Suppression(NMS) to select top-k bboxes arrcoding to the attn_score.
    keep_indices, count = delg_utils.nms(boxes = output_boxes,
                              scores = output_scores,
                              overlap = IOU_THRES,
                              top_k = TOP_K)
    keep_indices = keep_indices[:TOP_K]

    output_boxes = torch.index_select(output_boxes, dim=0, index=keep_indices)
    output_features = torch.index_select(output_features, dim=0, index=keep_indices)
    output_scales = torch.index_select(output_scales, dim=0, index=keep_indices)
    output_scores = torch.index_select(output_scores, dim=0, index=keep_indices)
    output_locations = delg_utils.CalculateKeypointCenters(output_boxes)
    
    data = {
        'locations':to_numpy(output_locations),
        'descriptors':to_numpy(output_features),
        'scores':to_numpy(output_scores)
        #'attention':to_numpy(output_original_scale_attn)
        }
    return data



# https://github.com/feymanpriv/DELG
if __name__ == "__main__":
  img_path = r"E:\datasets\bop_datasets\ycbv\crop_test\20\img\000048_000019.png"

  MODEL_WEIGHTS = ""
  SCALE_LIST = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0]
  _MEAN = [0.406, 0.456, 0.485]
  _SD = [0.225, 0.224, 0.229]
  RF = 291.0
  STRIDE = 16.0
  PADDING = 145.0
  ATTN_THRES = 260.
  IOU_THRES = 0.98
  ATTN_THRES = 260.0
  TOP_K = 1000

  cache_urls = True

  cfg.merge_from_file(r'C:\Users\DELL\Desktop\projs\BaseMethod_PY\method\delg\resnet_delg_8gpu.yaml')
  err_str = "The first lr step must start at 0"
  assert not cfg.OPTIM.STEPS or cfg.OPTIM.STEPS[0] == 0, err_str
  #data_splits = ["train", "val", "test"]
  #err_str = "Data split '{}' not supported"
  #assert _C.TRAIN.SPLIT in data_splits, err_str.format(_C.TRAIN.SPLIT)
  #assert _C.TEST.SPLIT in data_splits, err_str.format(_C.TEST.SPLIT)
  err_str = "Mini-batch size should be a multiple of NUM_GPUS."
  assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0, err_str
  assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0, err_str
  err_str = "Precise BN stats computation not verified for > 1 GPU"
  assert not cfg.BN.USE_PRECISE_STATS or cfg.NUM_GPUS == 1, err_str
  err_str = "Log destination '{}' not supported"
  assert cfg.LOG_DEST in ["stdout", "file"], err_str.format(cfg.LOG_DEST)
  if cache_urls:
      config.cache_cfg_urls()
  
  cfg.freeze()
  total_card = cfg.INFER.TOTAL_NUM
  assert total_card > 0, 'cfg.TOTAL_NUM should larger than 0. ~'
  assert cfg.INFER.CUT_NUM <= total_card, "cfg.CUT_NUM <= cfg.TOTAL_NUM. ~"

  model = setup_model()
  im = cv2.imread(img_path)
  im = im.astype(np.float32, copy=False)
  data =  delg_extract(im, model)

  xys = data["locations"]

  img = cv2.imread(img_path)
  kps = []
  for x, y, scale in xys:
    print(x, y, scale)
    kp = cv2.KeyPoint(x, y, 0)
    kps.append(kp)
  img_keypoints = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
  cv2.drawKeypoints(img, kps, img_keypoints)
  #cv2.imwrite("keynet.png", img_keypoints)
  cv2.imshow("keynet", img_keypoints)
  cv2.waitKey(0)
  print()