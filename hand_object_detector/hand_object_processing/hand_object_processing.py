import iou
import detectron2demo
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hand_object_demo

person_to_run = 'P99' #set to empty string to run all
subsample = 1

print("Running detectron")
#detectron2demo.run(person_to_run, subsample)
os.chdir('..') #Fixes some directory dependencies of hand_object_demo
print("Running hand-object detection")
#hand_object_demo.run(person_to_run, subsample)
os.chdir('./hand_object_processing')
print("Running IOU check")
iou.processData(person_to_run, subsample)