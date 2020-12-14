import numpy as np
import os
import time
import pycocotools.mask as mask_util
from detectron2.utils.colormap import random_color
import random, string
import matplotlib.pyplot as plt

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

def processData(person_to_run, subsample):
    for root, directories, filenames in os.walk(mask_rcnn_path): 
        video_visualizer = VideoVisualizer() #new visulizer instance for each directory (video)
        video_visualizer_hands = VideoVisualizer()
        objIDlist = []
        handIDlist = []
        detectedList = []
        detectedHandsList = []
        seriesList = []
        timesList = []
        objIDsList = []
        objBBsList = []
        
        #if root.endswith("P01_107"):
        if person_to_run in root:
            for filename in sorted(filenames):
                if filename.endswith(".npz") and 'meta' not in filename:# and int(filename[6:-4])<500):
                    frameNum = int(filename.split('.')[0][6:])
                    frameInd = frameNum // subsample
                    path = os.path.join(root,filename)
                    pathList = path.split('/')
                    pathList[pathList.index("maskrcnn_det")] = "images_det"
                    doh_path = "/".join(pathList[:-1]) + "/"
                    print(doh_path + filename.split('.')[0] + '.npz')

                    if os.path.isfile(doh_path + filename.split('.')[0] + '.npz'):
                        print(path)
                        data = np.load(path)
                        maskrcnn_boxes = data['boxes']
                        maskrcnn_labels = data['classes']
                        label_list = data['label'].tolist()
                        maskrcnn_boxes = maskrcnn_boxes.reshape(-1,4)
                        doh_data = np.load(doh_path + filename.split('.')[0] + '.npz', allow_pickle=True)
                        doh_boxes = doh_data['objects']
                        doh_hands = doh_data['hands']

                        contact = checkContact(doh_boxes, maskrcnn_boxes)

                        detected = [
                            _DetectedInstance(maskrcnn_labels[i], maskrcnn_boxes[i], mask_rle=None, color=None, ttl=8, objID=None, contact=contact[:,i])
                            for i in range(len(maskrcnn_labels))
                        ]
                        if doh_hands.size > 1:
                            detectedHands = [
                                _DetectedInstance('hand', doh_hands[i], mask_rle=None, color=None, ttl=8, objID=None, contact=None)
                                for i in range(doh_hands.shape[0])
                            ]
                        else:
                            detectedHands = []
                        
                        objIDs = video_visualizer.assign_ids(detected)
                        handIDs = video_visualizer_hands.assign_ids(detectedHands)
                        objIDlist.append(objIDs)
                        handIDlist.append(handIDs)
                        detectedList.append(detected)
                        detectedHandsList.append(detectedHands)

                        #hand2obj = imageStabilization(doh_hands, maskrcnn_boxes, os.path.join(doh_path + filename.split('.')[0] + "stab"), objIDs)
                        #findContactLabel(doh_boxes, doh_hands, maskrcnn_boxes, os.path.join(doh_path + filename + ".txt"), label_list, maskrcnn_labels, objIDs)
                    else:
                        print("No DOH file found for" + path)
                    
                    series_refresh_rate = 1 #Defines a new series every x frames
                    series_length = 20
                    if len(objIDlist) > series_length and len(detectedHands) > 0 and frameInd % series_refresh_rate == 0:
                        if objIDlist[-1] is not None:
                            objLabels = [d.label for d in detectedList[-1]]
                            seriesObjs = []
                            handObjs = []

                            #Find all objects that are valid for 20 frames
                            for objID in objIDlist[-1]: #loop through objects in most recent frame
                                #Do the last 20 frames have objID in them?
                                checkArray = np.zeros(series_length)
                                series = np.zeros(series_length)
                                for i in range(series_length): 
                                    if objIDlist[-i-1] is None:
                                        checkArray[i] = False
                                    else:
                                        checkArray[i] = objID in objIDlist[-i-1]
                                        #series[i] = 
                                #print(checkArray)
                                if np.all(checkArray):
                                    if label_list[objLabels[objIDlist[-1].index(objID)]] != 'person':
                                        seriesObjs.append(objID) #Gather all objects that are valid for 20 frames

                            #Find all hands that are valid for 20 frames
                            for handID in handIDlist[-1]: #loop through objects in most recent frame
                                #Do the last 20 frames have objID in them?
                                checkArray = np.zeros(series_length)
                                series = np.zeros(series_length)
                                for i in range(series_length): 
                                    if handIDlist[-i-1] is None:
                                        checkArray[i] = False
                                    else:
                                        checkArray[i] = handID in handIDlist[-i-1]
                                        #series[i] = 
                                #print(checkArray)
                                if np.all(checkArray):
                                    handObjs.append(handID) #Gather all objects that are valid for 20 frames

                            #Build series from valid objects
                            for handID in handObjs:
                                for objID in seriesObjs:
                                    series = np.zeros((20,3))
                                    for i in range(series_length):
                                        objInd = objIDlist[-i-1].index(objID)
                                        #print(objInd)
                                        handInd = handIDlist[-i-1].index(handID)
                                        objBboxes = [d.bbox for d in detectedList[-i-1]]
                                        objContact = np.array([d.contact for d in detectedList[-i-1]]).T
                                        handBboxes = [d.bbox for d in detectedHandsList[-i-1]]

                                        series[i,0:2] = np.subtract(getCentroid(objBboxes[objInd]), getCentroid(handBboxes[handInd]))
                                        series[i,2] = objContact[handInd, objInd]
                                    if np.count_nonzero(series) > 0:
                                        seriesList.append(series)
                                        timesList.append(frameNum)
                                        objIDsList.append(objID)
                                        objInd = objIDlist[-5].index(objID)
                                        objBBsList.append([d.bbox for d in detectedList[-5]][objInd])
                                        #Visualize hand-object tracking
                                        #plt.plot(series[:,0], series[:,1])
                                        #plt.scatter(series[:,0], series[:,1], c=series[:,2])
                                        #plt.show()
            if len(seriesList) > 0:
                np.save(os.path.join(root + "stab"), seriesList)
                np.savez(os.path.join(root + "meta"), times = timesList, handsList=detectedHandsList, objList=detected, objIDs=objIDsList, objBBsPrev=objBBsList)
                

def imageStabilization(doh_hands, maskrcnn_boxes, save_path, objIDs):
    if doh_hands.size > 1:
        hand2obj = np.zeros((doh_hands.shape[0], maskrcnn_boxes.shape[0], 2))
        for j,doh_box in enumerate(doh_hands):
            for i,maskrcnn_box in enumerate(maskrcnn_boxes):
                hand2obj[j,i,:] = np.subtract(getCentroid(maskrcnn_box), getCentroid(doh_box))
    else:
        hand2obj = None
    np.save(save_path, hand2obj)
    return hand2obj

def checkContact(doh_boxes, maskrcnn_boxes):
    """
    Cross-checks IoU for doh_boxes and maskrcnn_boxes. If m=doh_boxes.shape[0] and
    n = maskrcnn_boxes.shape[0], output is m*n array that indicates which boxes are
    being contacted.
    """
    if doh_boxes.size < 2:
        return np.zeros((1, maskrcnn_boxes.shape[0]))
    
    contact = np.zeros((doh_boxes.shape[0], maskrcnn_boxes.shape[0]))
    for j,doh_box in enumerate(doh_boxes):
        max_iou = -1
        bestInd = -1
        for i,maskrcnn_box in enumerate(maskrcnn_boxes):
            val = bb_intersection_over_union(doh_box,maskrcnn_box)
            if (val > max_iou):
                bestInd = i
                max_iou = val
        if max_iou > IOU_THRESHOLD:
            contact[j, bestInd] = 1

    return contact



def findContactLabel(doh_boxes, doh_hands, maskrcnn_boxes, save_path, label_list, labels, objIDs):
    """
    Cross-checks IoU for doh_boxes and maskrcnn_boxes, and saves a text file with the matching label, if found
    Also prints the location of detected hands to the text file
    """
    outputFile = open(save_path, 'w')
    for j,doh_box in enumerate(doh_boxes):
        max_iou = -1
        bestBox = 0
        bestID = 0
        for i,maskrcnn_box in enumerate(maskrcnn_boxes):
        
            
            val = bb_intersection_over_union(doh_box,maskrcnn_box)
            if (val > max_iou):
                bestLabel = str(label_list[labels[i]])
                bestID = objIDs[i]
                bestBox = maskrcnn_box
                max_iou = val

        if (doh_boxes.shape[0] == 0):
            print("No_contact", file = outputFile, end=',')
            print("NaN,NaN,NaN,NaN", file = outputFile, end=',')
        elif (max_iou > IOU_THRESHOLD):
            cx, cy = getCentroid(bestBox)
            print(bestLabel, file = outputFile, end=',')
            print(bestID, file = outputFile, end=',')
            print(cx, file = outputFile, end=',')
            print(cy, file = outputFile, end=',')
            print(str(max_iou), file = outputFile)
        else:
            cx, cy = getCentroid(doh_box)
            print("unknown", file = outputFile, end=',')
            print("NaN", file = outputFile, end=',')
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
            
    def assign_ids(self, instances):
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
                    if instances[newidx].objID is None:
                        instances[newidx].objID = inst.objID
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
                if inst.objID is None:
                    #Assign random 32-bit hex key
                    inst.objID = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
            self._old_instances = instances[:] + extra_instances
            return [d.objID for d in instances]

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

    __slots__ = ["label", "bbox", "mask_rle", "color", "ttl", "objID", "contact"]

    def __init__(self, label, bbox, mask_rle, color, ttl, objID, contact):
        self.label = label
        self.bbox = bbox
        self.mask_rle = mask_rle
        self.color = color
        self.ttl = ttl
        self.objID = objID
        self.contact = contact

#checkIoU()