# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:48:41 2021

@author: tolga
"""

import os
import os.path as osp
import argparse
import numpy as np 
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm 
 
from coco_loader import coco_loader
from torchvision import models                                                                     
from convcap import convcap
from vggfeats import Vgg16Feats
from evaluate import language_eval
from nltk.translate.bleu_score import corpus_bleu
import pdb
 

def repeat_img_per_cap(imgsfeats, imgsfc7, ncap_per_img):

    batchsize, featdim, feat_h, feat_w = imgsfeats.size()
    batchsize_cap = batchsize*ncap_per_img
    imgsfeats = imgsfeats.unsqueeze(1).expand(\
      batchsize, ncap_per_img, featdim, feat_h, feat_w)
    imgsfeats = imgsfeats.contiguous().view(\
      batchsize_cap, featdim, feat_h, feat_w)
    
    batchsize, featdim = imgsfc7.size()
    batchsize_cap = batchsize*ncap_per_img
    imgsfc7 = imgsfc7.unsqueeze(1).expand(\
      batchsize, ncap_per_img, featdim)
    imgsfc7 = imgsfc7.contiguous().view(\
      batchsize_cap, featdim)
  
    return imgsfeats, imgsfc7


def test( model_dir, data_dir, batchsize, split, num_layers, attention, nthreads=0, modelfn=None, model_convcap=None, model_imgcnn=None):
  """Runs test on split=val/test with checkpoint file modelfn or loaded model_*"""
    
  
  DICT_SIZE = 7541  
  
  data_name = "coco_5_cap_per_img_5_min_word_freq"
  t_start = time.time()
  data = coco_loader( data_dir, data_name, split=split, max_tokens=15)
  print('[DEBUG] Loading %s data ... %f secs' % (split, time.time() - t_start))

  data_loader = DataLoader(dataset=data, num_workers=nthreads,\
    batch_size=batchsize, shuffle=False, drop_last=True)

  batchsize = batchsize
  max_tokens = data.max_tokens
  num_batches = np.int_(np.floor((len(data.ids)*1.)/batchsize))
  print('[DEBUG] Running inference on %s with %d batches' % (split, num_batches))

  if(modelfn is not None):
    model_imgcnn = Vgg16Feats()
    model_imgcnn.cuda() 

    model_convcap = convcap(data.numwords, num_layers, is_attention=attention)
    model_convcap.cuda()

    print('[DEBUG] Loading checkpoint %s' % modelfn)
    checkpoint = torch.load(modelfn)
    model_convcap.load_state_dict(checkpoint['state_dict'])
    model_imgcnn.load_state_dict(checkpoint['img_state_dict'])
  else:
    model_imgcnn = model_imgcnn
    model_convcap = model_convcap

  model_imgcnn.train(False) 
  model_convcap.train(False)

  pred_captions = []
  
  #Test epoch
  total_CEloss = 0
  for batch_idx, (imgs, captions, true_wordclass, _, img_ids) in \
    tqdm(enumerate(data_loader), total=num_batches):
    
    
    imgs = imgs.view(batchsize, 3, 224, 224)

    imgs_v = Variable(imgs.cuda())
    imgsfeats, imgsfc7 = model_imgcnn(imgs_v)
    _, featdim, feat_h, feat_w = imgsfeats.size()
  
    wordclass_feed = np.zeros((batchsize, max_tokens), dtype='int64')
    
    wordclass_feed[:,0] = 7539 #start token index
    
    outcaps = np.empty((batchsize, 0)).tolist()

    for j in range(max_tokens-1):
      wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()

      wordact, _ = model_convcap(imgsfeats, imgsfc7, wordclass)

      wordact = wordact[:,:,:-1]
      wordact_t = wordact.permute(0, 2, 1).contiguous().view(batchsize*(max_tokens-1), -1)
      
      
      wordprobs = F.softmax(wordact_t).cpu().data.numpy()
      wordids = np.argmax(wordprobs, axis=1)
      wordprobmax = np.max(wordprobs, axis=1)
      
      for k in range(batchsize):
         word = data.word_list[wordids[j+k*(max_tokens-1)]]
         outcaps[k].append(word)
         if(j < max_tokens-1):
           wordclass_feed[k, j+1] = wordids[j+k*(max_tokens-1)]
  
      
    for j in range(batchsize):
      num_words = len(outcaps[j]) 
      if '<end>' in outcaps[j]:
        num_words = outcaps[j].index('<end>')
      outcap = ' '.join(outcaps[j][:num_words])
      pred_captions.append({'image_id': img_ids[j], 'caption': outcap})
    
    #pdb.set_trace()

    # cross entropy ---------------------------
    batch_CEloss = 0
    for j in range(5): #number of captions per image in ground truth dataset
      OHE_captions = matrix_OHE(true_wordclass[:,j,:])
      batch_CEloss = batch_CEloss + calc_batch_CEloss(OHE_captions, wordprobs)/DICT_SIZE
    
    total_CEloss = total_CEloss + batch_CEloss/5
  total_CEloss = total_CEloss / num_batches
      
  # creating hypotheses
  list_pred_captions = []
  for i in range(len(pred_captions)):
    list_pred_captions.append(pred_captions[i]['caption'].split())


  true_captions, txt_true_captions = call_caption_text(data_dir, split)
  txt_true_captions = txt_true_captions[0:len(list_pred_captions)]
  BLEU_score = corpus_bleu( txt_true_captions , list_pred_captions)
  
  #pdb.set_trace()
  
  model_imgcnn.train(True) 
  model_convcap.train(True)

  return BLEU_score, total_CEloss 



def call_caption_text( data_folder, test_or_val ):
    
    
    data_name = "coco_5_cap_per_img_5_min_word_freq"
    with open(os.path.join(data_folder, test_or_val + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
        true_captions = torch.LongTensor(json.load(j))
    
    with open(data_folder + '/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json', 'r') as j:
        word_dict = json.load(j)
       
    numwords = len(word_dict)    
    word_list = []
    for key, value in word_dict.items():
        temp = [key,value]
        word_list.append(temp)
    
    upd_word_list = []
    upd_word_list.append( word_list[-1][0] )
    for i in range(0,numwords-1):
        upd_word_list.append( word_list[i][0] )
    #upd_word_list.append( self.word_list[-1][0] )
    word_list = upd_word_list
    
    VECTOR_SIZE = 52
    END_INDEX = 7540
    CAPTION_NUM = 5
    cap_len = len(true_captions)
    txt_true_captions = []
    
    for i in range(int(cap_len/CAPTION_NUM)):
        caption_batch = []
        for k in range(i*CAPTION_NUM, (i+1)*CAPTION_NUM):
            img_caption = []
            for j in range(1,VECTOR_SIZE):
                word_index = true_captions[k,j]
                if word_index != END_INDEX:
                    img_caption.append( word_list[word_index] )
                else:
                    break
            caption_batch.append(img_caption)
        txt_true_captions.append(caption_batch)
        
    return true_captions, txt_true_captions



def vector_map( x ):
    
    DICT_SIZE = 7541
    vector = np.zeros([DICT_SIZE])
    vector[x] = 1
    
    return vector

def matrix_OHE( x ): # take 20x15 matrix

    x = x.cpu().data.numpy()
    DICT_SIZE = 7541
    x = x[:,1:]
    x_h, x_w = x.shape
    new_h = x_h*x_w
    
    ohe = np.zeros([new_h, DICT_SIZE])
    
    for i in range(x_h):
        for j in range(x_w):
            ohe[i*x_w+j] = vector_map(x[i,j])
    
    return ohe

def calc_batch_CEloss( a, b):
    # a: OHE ground truth word indexes
    # b: predicted word probabilities
    
    #log_b = np.log(b)
    log_b = np.where(b > 0.0000000001, np.log(b), -10)
    return -np.sum(a*log_b)



if __name__ == '__main__':
    
    
    x = torch.tensor([[7539,    1, 2652,   17,  248,  166,   22,    1, 2289,   33, 7540,    0,
            0,    0,    0],
        [7539,  128,   33, 2285,  165,  311,   22, 2289,   33, 7540,    0,    0,
            0,    0,    0],
        [7539,    1,  296,   17,  222,   22,    1,   39,   33, 7540,    0,    0,
            0,    0,    0],
        [7539,    1,  313,   17,  126,   86,   61,   17,    1,   33, 7540,    0,
            0,    0,    0],
        [7539,   23,  248,   85,   31,   25,   23, 5491, 2391, 3033, 7540,    0,
            0,    0,    0],
        [7539,    1,   33, 2464, 2465, 1675, 2835,   22,   23,  405, 7540,    0,
            0,    0,    0],
        [7539,   23,   33,   26,   67,   17,  803,  804,  200, 7540,    0,    0,
            0,    0,    0],
        [7539,    1,   33,   59,    3,    1,  203,    7,  125,  147,   22,   18,
         7540,    0,    0],
        [7539,    1,    4,  155,   33,   11,  804, 4425,    7,  330, 7540,    0,
            0,    0,    0],
        [7539,    1,  803,  804,   33, 2222,   11,  200,    7,  330, 7540,    0,
            0,    0,    0],
        [7539,   94,  222,   85,  110,  155,   87,    1,  215, 1344, 7540,    0,
            0,    0,    0],
        [7539,   94,  222, 3604,  130,  233,  898,   87,    1, 1344, 7540,    0,
            0,    0,    0],
        [7539,    1,  296,   17,   88,  222,  218,   22,  246,   17,    1, 1447,
          215, 7540,    0],
        [7539,   94,  222, 4293,  166,   15,    1, 1569, 1344, 7540,    0,    0,
            0,    0,    0],
        [7539,    1, 1001,   17,  222,  155,   87,    1, 3457, 1025,   15,    1,
           10, 7540,    0],
        [7539,    1,   84,   26, 1375,    1,  396,   17,  408,   92,  143,    1,
           30, 7540,    0],
        [7539,    1,   84, 1375,    1,  396, 7119,  143,    1,   30, 7540,    0,
            0,    0,    0],
        [7539,   30,   26, 2071,    1, 2341,   17,   92,  132,    1,   84,   22,
          246,   17, 7540],
        [7539,    1,   84,  397,    1, 5067,   17,  408,  329,  143,    1,   30,
           22,  246, 7540],
        [7539,  128,  126,  390,    1,   84,  342,    1,   30,   92,   61,    1,
            2,   33, 7540]])
    
    
    #c = matrix_OHE(x)
    
    
    deneme = torch.tensor([[0, 0],[0, 1]])
    probs = np.array([[1/2,1/2],[0.2, 0.8]])
    
    OHE_deneme = matrix_OHE(deneme)
    
    log_b = np.where(probs > 0.0000000001, np.log(probs), -10)
    
    res = OHE_deneme * log_b
    
    
    
    
    
    
    
    





