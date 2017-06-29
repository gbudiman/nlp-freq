from src.CocoUtil import CocoUtil

coco_util = CocoUtil()
coco_util.get_coco_objects('data/instances_train2014.json', out='dump/objects_train.json')
coco_util.get_coco_objects('data/instances_val2014.json', out='dump/objects_val.json')