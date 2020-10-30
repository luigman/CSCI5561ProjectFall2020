import os.path as osp
import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

#this_dir = osp.dirname(__file__)
this_dir = os.path.dirname(os.getcwd()) #Use parent directory

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

coco_path = osp.join(this_dir, 'data', 'coco', 'PythonAPI')
add_path(coco_path)
