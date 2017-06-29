from src.CocoUtil import CocoUtil

coco_util = CocoUtil()
#coco_util.sample_annotations()
coco_util.build_dataset(get='nouns', out='dump/nouns.txt', stripped_json_path='dump/captions_fixed.json')