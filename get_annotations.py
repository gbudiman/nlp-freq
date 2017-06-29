from src.CocoUtil import CocoUtil

coco_util = CocoUtil()
#coco_util.sample_annotations()
coco_util.dump_annotations('dump/annotations.json')