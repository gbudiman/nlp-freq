import pprint as pp
import json
import operator
import re
import spacy
from tqdm import tqdm

class Analytic:
  nlp = None
  raw_data = None
  result = None
  merge_rule = None
  gut_color_rule = None
  mode = None


  def __init__(self, mode='train'):
    self.mode = mode
    self.nlp = spacy.load('en')
    path = 'data/captions_train2014.json' if self.mode == 'train' else 'data/captions_val2014.json'
    with open(path) as json_file:
      self.raw_data = json.load(json_file)

  def nlp_action(self, word_input):
    doc = self.nlp(word_input)
    smart_insert    = self.smart_insert
    merge_rule      = self.merge_rule
    gut_color_rule  = self.gut_color_rule
    writebacks = []

    for word in doc:
      wtag = word.tag_
      #wlemma = word.lemma_
      wlemma = word.text

      if (wtag.startswith('NN')):
        if merge_rule is not None and word.text in merge_rule:
          wlemma = merge_rule[word.text]

        smart_insert('nouns', wlemma)
      elif (wtag.startswith('VB')):
        smart_insert('verbs', wlemma)
      elif (wtag.startswith('JJ')):
        if gut_color_rule is not None and word.text in gut_color_rule:
          wlemma = None
        else:
          smart_insert('adjs', wlemma)
      elif (wtag.startswith('RB')):
        smart_insert('advs', wlemma)

      if wlemma is not None: writebacks.append(wlemma)

    return writebacks

  def smart_insert(self, wtype, lemma):
    entry = self.result[wtype]
    entry_exist = lemma in entry

    if not entry_exist:
      self.result[wtype][lemma] = 1
    else:
      self.result[wtype][lemma] = entry[lemma] + 1

  def build_dataset(self, limit=-1, writeback=False):
    self.result = {
      'nouns': {},
      'verbs': {},
      'adjs': {},
      'advs': {}
    }

    wb_builder = []

    input_data = self.raw_data['annotations'][0:limit]
    for annotation in tqdm(input_data):
      wb = self.nlp_action(annotation['caption'])

      if writeback:
        coco_internal_id = annotation['id']
        image_id = annotation['image_id']
        s = ' '.join(wb)
        
        wb_builder.append({
          'id': coco_internal_id,
          'image_id': image_id,
          'caption': s
        })

    if writeback:
      out_path = 'dump/merged_train_annotations.json' if self.mode == 'train' else 'dump/merged_val_annotations.json'
      with open(out_path, 'w') as outfile:
        json.dump(wb_builder, outfile, indent=2)

  def analyze(self, t):
    cue = self.result[t]
    sorted_freq = sorted(cue.items(), key=operator.itemgetter(1))

    for word in sorted_freq:
      print('{1:5d} {0:s}').format(word[0], word[1])

  def merge_topics(self):
    self.load_merge_rule()
    self.build_dataset(writeback = True)
    self.analyze('nouns')
    self.analyze('adjs')

  def load_merge_rule(self):
    self.merge_rule = {}
    self.gut_color_rule = {}

    nouns_path = 'dump/train_nouns_topic_master.txt' if self.mode == 'train' else 'dump/val_nouns_topic_master.txt'
    adjs_path = 'dump/train_adjs_topic_master.txt' if self.mode == 'train' else 'dump/val_adjs_topic_master.txt'

    with open(nouns_path) as f:
      for line in f.readlines():
        matched = re.search(r'\s*\d+\s+(\w+)\s+(\w+)', line)
        if matched:
          #print('{} -> {}').format(matched.group(1), matched.group(2))
          self.merge_rule[matched.group(1)] = matched.group(2)

    with open(adjs_path) as f:
      for line in f.readlines():
        matched = re.search(r'\s*\d+\s+(\w+)\s+\*', line)
        if matched:
          self.gut_color_rule[matched.group(1)] = True

    