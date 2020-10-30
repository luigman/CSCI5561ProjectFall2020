import numpy as np
import os
import time

IOU_THRESHOLD = 0.2

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def check_boxes(bbox1,bbox2):
    x1,y1,x2,y2 = bbox1
    x21,y21,x22,y22 = bbox2

    if x1 >=x21 and y1>=y21:
        if x2 <= x22 and y2 <= y22:
            return True
    
    return False

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
doh_path = os.path.join(parent_dir, "images_det/npy/")
mask_rcnn_path = os.path.join(parent_dir, "maskrcnn_det/npy/")

def checkIoU():
    file_list = os.listdir(mask_rcnn_path)
    for filename in file_list:
        data = np.load(mask_rcnn_path + filename)
        maskrcnn_boxes = data['boxes']
        labels = data['classes']
        label_list = data['label'].tolist()
        maskrcnn_boxes = maskrcnn_boxes.reshape(-1,4)
        doh_boxes = np.load(doh_path + filename.split('.')[0] + '.npy').reshape(-1,4)
        max_iou = -1
        pred_label = None
        for i,maskrcnn_box in enumerate(maskrcnn_boxes):
            for j,doh_box in enumerate(doh_boxes):
                """
                iou  = check_boxes(doh_box,maskrcnn_box)
                if iou > max_iou:
                    max_iou = iou
                    pred_label = label_list[labels[i]]
                """
                val = bb_intersection_over_union(doh_box,maskrcnn_box)
                if (val > max_iou):
                    bestLabel = str(label_list[labels[i]])
                    max_iou = val
        outputFile = open(os.path.join(parent_dir, "images_det/" + filename + ".txt"), 'w')
        if (doh_boxes.shape[0] == 0):
            print("No contact was made in file " + filename, file = outputFile)
        elif (max_iou > IOU_THRESHOLD):
            print("Class: " + bestLabel, file = outputFile)
            print("IoU: " + str(max_iou), file = outputFile)
        else:
            print("An unknown object was contacted in file " + filename, file = outputFile)

        outputFile.close()

checkIoU()