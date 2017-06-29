import pprint as pp
import json
import re
from tqdm import tqdm
import spacy

class CocoUtil:
  raw_data = None
  nlp = None

  def __init__(self):
    self.nlp = spacy.load('en')
    with open('data/captions_train2014.json') as json_file:
      self.raw_data = json.load(json_file)

  def sample_annotations(self):
    for annotation in self.raw_data['annotations'][0:8]:
      pp.pprint(annotation)

  def dump_annotations(self, file_path, limit=-1, fix_dict=None):
    t = {}

    for annotation in tqdm(self.raw_data['annotations'][0:limit]):
      image_id = int(annotation['image_id'])
      caption = annotation['caption']

      if image_id not in t:
        t[image_id] = []

      if fix_dict is not None:
        granular = []
        splits = re.split(r'\s+', caption)
        for _word in splits:
          word = _word.lower()

          unpunctuated_match = re.search(r'(\w+)', word)
          if unpunctuated_match:
            unpunctuated_word = unpunctuated_match.group(1)
            if unpunctuated_word in fix_dict:
              granular.append(re.sub(unpunctuated_word, fix_dict[unpunctuated_word], word))
            else:
              granular.append(word)
          else:
            granular.append(word)

        t[image_id].append(' '.join(granular)) 
      else:
        t[image_id].append(caption)

    #pp.pprint(t)
    with open(file_path, 'w') as outfile:
      json.dump(t, outfile, indent=2, sort_keys=True)

  def build_dataset(self, limit = -1, get=None, out=None, stripped_json_path=None):
    result = {
      'nouns': {},
      'verbs': {},
      'adjs': {},
      'advs': {}
    }

    def smart_insert(wtype, lemma):
      entry = result[wtype]
      entry_exist = lemma in entry

      if not entry_exist:
        result[wtype][lemma] = 1
      else:
        result[wtype][lemma] = entry[lemma] + 1

    def nlp_action(word_input):
      doc = self.nlp(word_input)

      for word in doc:
        wtag = word.tag_
        wlemma = word.lemma_
        if (wtag.startswith('NN')):
          smart_insert('nouns', wlemma)
        elif (wtag.startswith('VB')):
          smart_insert('verbs', wlemma)
        elif (wtag.startswith('JJ')):
          smart_insert('adjs', wlemma)
        elif (wtag.startswith('RB')):
          smart_insert('advs', wlemma)

    if stripped_json_path is None:
      input_data = self.raw_data['annotations'][0:limit]

      for annotation in tqdm(input_data):
        caption = annotation['caption']
        nlp_action(caption)
        
    else:
      input_data = None
      with open(stripped_json_path) as f:
        input_data = json.load(f)

      for coco_id in tqdm(input_data):
        captions = input_data[coco_id]
        for caption in captions:
          nlp_action(caption)



    if out is not None:
      s = ''
      wordkeys = sorted(result[get].keys());
      # for word, freq in result[get].iteritems():
      #   s += word + ': ' + str(freq) + "\n"

      for word in wordkeys:
        s += word + ': ' + str(result[get][word]) + "\n"

      with open(out, 'w') as outfile:
        outfile.write(s)

    else:
      return result

  def build_fixer(self, path):
    content = None
    pattern = r'(\w+)\:\s\d+\s(.+)'
    fix_dict = {}
    with open(path) as f:
      for line in f:
        stripped_line = line.strip()
        match = re.search(pattern, stripped_line)

        if match:
          fix_dict[match.group(1).lower()] = match.group(2).lower()
    
    return fix_dict
