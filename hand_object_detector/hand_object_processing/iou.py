import numpy as np
import os
import time
import pycocotools.mask as mask_util
from detectron2.utils.colormap import random_color

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

def processData():
    for root, directories, filenames in os.walk(mask_rcnn_path): 
        video_visualizer = VideoVisualizer() #new visulizer instance for each directory (video)
        for filename in sorted(filenames):
            if (filename.endswith(".npz")):
                path = os.path.join(root,filename)
                print(path)
                pathList = path.split('/')
                pathList[pathList.index("maskrcnn_det")] = "images_det"
                doh_path = "/".join(pathList[:-1]) + "/"

                data = np.load(path)
                maskrcnn_boxes = data['boxes']
                maskrcnn_labels = data['classes']
                label_list = data['label'].tolist()
                maskrcnn_boxes = maskrcnn_boxes.reshape(-1,4)
                doh_data = np.load(doh_path + filename.split('.')[0] + '.npz', allow_pickle=True)
                doh_boxes = doh_data['objects']
                doh_hands = doh_data['hands']

                #imageStabilization(doh_hands, maskrcnn_boxes, os.path.join(doh_path + filename.split('.')[0] + "stab"))
                #findContactLabel(doh_boxes, doh_hands, maskrcnn_boxes, os.path.join(doh_path + filename + ".txt"), label_list, maskrcnn_labels)
                detected = [
                    _DetectedInstance(maskrcnn_labels[i], maskrcnn_boxes[i], mask_rle=None, color=None, ttl=8)
                    for i in range(len(maskrcnn_labels))
                ]
                
                video_visualizer.assign_colors(detected)



                

def imageStabilization(doh_hands, maskrcnn_boxes, save_path):
    if doh_hands.size > 1:
        hand2obj = np.zeros((doh_hands.shape[0], maskrcnn_boxes.shape[0], 2))
        for j,doh_box in enumerate(doh_hands):
            for i,maskrcnn_box in enumerate(maskrcnn_boxes):
                hand2obj[j,i,:] = np.subtract(getCentroid(maskrcnn_box), getCentroid(doh_box))
    else:
        hand2obj = None
    np.save(save_path, hand2obj)
    return

def findContactLabel(doh_boxes, doh_hands, maskrcnn_boxes, save_path, label_list, labels):
    """
    Cross-checks IoU for doh_boxes and maskrcnn_boxes, and saves a text file with the matching label, if found
    Also prints the location of detected hands to the text file
    """
    outputFile = open(save_path, 'w')
    for j,doh_box in enumerate(doh_boxes):
        max_iou = -1
        bestBox = 0
        for i,maskrcnn_box in enumerate(maskrcnn_boxes):
        
            
            val = bb_intersection_over_union(doh_box,maskrcnn_box)
            if (val > max_iou):
                bestLabel = str(label_list[labels[i]])
                bestBox = maskrcnn_box
                max_iou = val

        if (doh_boxes.shape[0] == 0):
            print("No_contact", file = outputFile, end=',')
            print("NaN,NaN,NaN", file = outputFile, end=',')
        elif (max_iou > IOU_THRESHOLD):
            cx, cy = getCentroid(bestBox)
            print(bestLabel, file = outputFile, end=',')
            print(cx, file = outputFile, end=',')
            print(cy, file = outputFile, end=',')
            print(str(max_iou), file = outputFile)
        else:
            cx, cy = getCentroid(doh_box)
            print("unknown", file = outputFile, end=',')
            print(cx, file = outputFile, end=',')
            print(cy, file = outputFile, end=',')
            print("NaN", file = outputFile)

    print("", file = outputFile)
    if doh_hands.size > 1:
        for i, handbox in enumerate(doh_hands):
            cx, cy = getCentroid(handbox)
            print(cx, file = outputFile, end=',')
            print(cy, file = outputFile)

    outputFile.close()

class VideoVisualizer:
    def __init__(self):
        self._old_instances = []
            
    def assign_colors(self, instances):
            """
            Modified from detectron2/utils/video_visualizer.py

            Naive tracking heuristics to assign same color to the same instance,
            will update the internal state of tracked instances.

            Returns:
                list[tuple[float]]: list of colors.
            """
            if len(instances) == 0:
                return #no new detections to check

            # Compute iou with either boxes or masks:
            is_crowd = np.zeros((len(instances),), dtype=np.bool)
            if instances[0].bbox is None:
                assert instances[0].mask_rle is not None
                # use mask iou only when box iou is None
                # because box seems good enough
                rles_old = [x.mask_rle for x in self._old_instances]
                rles_new = [x.mask_rle for x in instances]
                ious = mask_util.iou(rles_old, rles_new, is_crowd)
                threshold = 0.5
            else:
                boxes_old = [x.bbox for x in self._old_instances]
                boxes_new = [x.bbox for x in instances]
                ious = mask_util.iou(boxes_old, boxes_new, is_crowd)
                threshold = 0.6
            if len(ious) == 0:
                ious = np.zeros((len(self._old_instances), len(instances)), dtype="float32")

            # Only allow matching instances of the same label:
            for old_idx, old in enumerate(self._old_instances):
                for new_idx, new in enumerate(instances):
                    if old.label != new.label:
                        ious[old_idx, new_idx] = 0

            matched_new_per_old = np.asarray(ious).argmax(axis=1)
            max_iou_per_old = np.asarray(ious).max(axis=1)

            # Try to find match for each old instance:
            extra_instances = []
            for idx, inst in enumerate(self._old_instances):
                if max_iou_per_old[idx] > threshold:
                    newidx = matched_new_per_old[idx]
                    if instances[newidx].color is None:
                        instances[newidx].color = inst.color
                        continue
                # If an old instance does not match any new instances,
                # keep it for the next frame in case it is just missed by the detector
                inst.ttl -= 1
                if inst.ttl > 0:
                    extra_instances.append(inst)

            # Assign random color to newly-detected instances:
            for inst in instances:
                if inst.color is None:
                    inst.color = random_color(rgb=True, maximum=1)
            self._old_instances = instances[:] + extra_instances
            return [d.color for d in instances]

class _DetectedInstance:
    """
    From detectron2/utils/video_visualizer.py

    Used to store data about detected objects in video frame,
    in order to transfer color to objects in the future frames.

    Attributes:
        label (int):
        bbox (tuple[float]):
        mask_rle (dict):
        color (tuple[float]): RGB colors in range (0, 1)
        ttl (int): time-to-live for the instance. For example, if ttl=2,
            the instance color can be transferred to objects in the next two frames.
    """

    __slots__ = ["label", "bbox", "mask_rle", "color", "ttl"]

    def __init__(self, label, bbox, mask_rle, color, ttl):
        self.label = label
        self.bbox = bbox
        self.mask_rle = mask_rle
        self.color = color
        self.ttl = ttl

#checkIoU()