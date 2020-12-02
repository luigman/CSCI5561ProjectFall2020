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

def getCentroid(box):
    x1, y1, x2, y2 = box
    return (x1+x2)/2, (y1+y2)/2

def check_boxes(bbox1,bbox2):
    x1,y1,x2,y2 = bbox1
    x21,y21,x22,y22 = bbox2

    if x1 >=x21 and y1>=y21:
        if x2 <= x22 and y2 <= y22:
            return True
    
    return False

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
#doh_path = os.path.join(parent_dir, "images_det/")
mask_rcnn_path = os.path.join(parent_dir, "maskrcnn_det/")

def checkIoU():
    for root, directories, filenames in os.walk(mask_rcnn_path): 
        for filename in filenames:
            if (filename.endswith(".npz")):
                path = os.path.join(root,filename)
                print(path)
                pathList = path.split('/')
                pathList[pathList.index("maskrcnn_det")] = "images_det"
                doh_path = "/".join(pathList[:-1]) + "/"

                data = np.load(path)
                maskrcnn_boxes = data['boxes']
                labels = data['classes']
                label_list = data['label'].tolist()
                maskrcnn_boxes = maskrcnn_boxes.reshape(-1,4)
                doh_data = np.load(doh_path + filename.split('.')[0] + '.npz', allow_pickle=True)
                doh_boxes = doh_data['objects']
                doh_hands = doh_data['hands']

                outputFile = open(os.path.join(doh_path + filename + ".txt"), 'w')
                for j,doh_box in enumerate(doh_boxes):
                    max_iou = -1
                    for i,maskrcnn_box in enumerate(maskrcnn_boxes):
                    
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

                    if (doh_boxes.shape[0] == 0):
                        print("No_contact", file = outputFile, end=',')
                        print("NaN", file = outputFile)
                    elif (max_iou > IOU_THRESHOLD):
                        print(bestLabel, file = outputFile, end=',')
                        print(str(max_iou), file = outputFile)
                    else:
                        print("unknown", file = outputFile, end=',')
                        print("NaN", file = outputFile)

                print("", file = outputFile)
                if doh_hands.size > 1:
                    for i, handbox in enumerate(doh_hands):
                        print(getCentroid(handbox), file = outputFile)

                outputFile.close()

#checkIoU()