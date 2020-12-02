import iou
import detectron2demo
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hand_object_demo

print("Running detectron")
#detectron2demo.run()
os.chdir('..') #Fixes some directory dependencies of hand_object_demo
print("Running hand-object detection")
#hand_object_demo.run()
os.chdir('./hand_object_processing')
print("Running IOU check")
iou.checkIoU()