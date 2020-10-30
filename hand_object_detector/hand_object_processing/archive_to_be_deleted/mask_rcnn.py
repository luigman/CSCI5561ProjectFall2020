import matplotlib.pyplot as plt
from gluoncv import model_zoo, data, utils
import numpy as np
import os
import cv2

image_path = '/home/aditya/hand_object_detector/images/'
save_path = '/home/aditya/hand_object_detector/maskrcnn_det/npy/'
det_image_save_path = '/home/aditya/hand_object_detector/maskrcnn_det/'
net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
file_list = os.listdir(image_path)
for filename in file_list:
    x, orig_img = data.transforms.presets.rcnn.load_test(image_path + filename)
    ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]
    final_bboxes = bboxes[np.argwhere(scores>0.5)[:,0]]
    to_save_name = filename.split('.')[0]
    np.save(save_path + to_save_name + '.npy',final_bboxes)
    # paint segmentation mask on images directly
    width, height = orig_img.shape[1], orig_img.shape[0]
    masks, _ = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
    orig_img = utils.viz.plot_mask(orig_img, masks)

    # identical to Faster RCNN object detection
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                            class_names=net.classes, ax=ax)
    fig.savefig(det_image_save_path + to_save_name + '.png', dpi=fig.dpi)



