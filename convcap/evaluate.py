"""
From Karpathy's neuraltalk2:
https://github.com/karpathy/neuraltalk2
Specifically:
https://github.com/karpathy/neuraltalk2/blob/master/coco-caption/myeval.py
"""

import sys
sys.path.insert(0, 'third_party/coco-caption')

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import os.path as osp
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import sys


def language_eval(input_data, savedir, split):
  if type(input_data) == str: # Filename given.
    checkpoint = json.load(open(input_data, 'r'))
    preds = checkpoint
  elif type(input_data) == list: # Direct predictions give.
    preds = input_data

  annFile = 'third_party/coco-caption/annotations/captions_val2014.json'
  coco = COCO(annFile)
  valids = coco.getImgIds()

  # Filter results to only those in MSCOCO validation set (will be about a third)
  preds_filt = [p for p in preds if p['image_id'] in valids]
  print ('Using %d/%d predictions' % (len(preds_filt), len(preds)))
  resFile = osp.join(savedir, 'result_%s.json' % (split))
  json.dump(preds_filt, open(resFile, 'w')) # Serialize to temporary json file. Sigh, COCO API...

  cocoRes = coco.loadRes(resFile)
  cocoEval = COCOEvalCap(coco, cocoRes)
  cocoEval.params['image_id'] = cocoRes.getImgIds()
  cocoEval.evaluate()

  # Create output dictionary.
  out = {}
  for metric, score in cocoEval.eval.items():
    out[metric] = score

  # Return aggregate and per image score.
  return out, cocoEval.evalImgs


if __name__ == '__main__':
    
  loss = nn.CrossEntropyLoss()
  p1 = torch.randn(3, 5)
  p2 = torch.empty(3, dtype=torch.long).random_(5)
  #output = loss(p1, p2)
  # print(output.data)
  # output.backward()
  
  p1 = []
  
  p1 = torch.FloatTensor([[0.4, 0.6],[0.1, 0.9]])
  p2 = torch.LongTensor([1,1])
  
  out2 = F.nll_loss(p1, p2)
  
  
  
  
  
  
  
  
  
  
  
  