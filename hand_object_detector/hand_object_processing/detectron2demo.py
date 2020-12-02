
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
import  matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
images_path = os.path.join(parent_dir, "images/")
save_path = os.path.join(parent_dir, "maskrcnn_det/")
images_list = os.listdir(images_path)
#im = cv2.imread("./test4.jpg")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

"""
Uncomment to enable visualizations


# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("frame",out.get_image()[:, :, ::-1])
cv2.waitKey(0)
"""
subsample = 1000
def run():
  for root, directories, filenames in os.walk(images_path): 
    for filename in filenames:  
      path = os.path.join(root,filename)
    
      if (not path.endswith(".jpg")):
        print("Not a jpg. Skipping...")
      elif int(filename[6:-4]) % subsample == 0:
        print(path)
        pathList = path.split('/')
        pathList[pathList.index("images")] = "maskrcnn_det"
        save_path = "/".join(pathList[:-1]) + "/"

        im = cv2.imread(path)
        outputs = predictor(im)
        class_preds = outputs["instances"].pred_classes.cpu().numpy()
        boxes_preds = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
        labels = v.metadata.get("thing_classes",None)
        if not(os.path.exists(save_path)):
          os.makedirs(save_path)
        np.savez(save_path + filename.split('.')[0],classes=class_preds,boxes=boxes_preds,label=np.array(labels))
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(save_path, filename),out.get_image()[:, :, ::-1])
        print("Saved: ", save_path)

