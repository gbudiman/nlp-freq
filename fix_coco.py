from src.CocoUtil import CocoUtil

coco_util = CocoUtil()
fix_dict = coco_util.build_fixer('data/coco_fixer.txt')
coco_util.dump_annotations('dump/captions_fixed.json', fix_dict=fix_dict)