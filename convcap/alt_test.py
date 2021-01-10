# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 21:53:07 2021

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


def test(model_dir, data_dir, batchsize, split, nthreads, attention=False, model_convcap=None, model_imgcnn=None):
    
    
    ncap_per_img = 5
    DICT_SIZE = 7541  
  
    data_name = "coco_5_cap_per_img_5_min_word_freq"
    t_start = time.time()
    data = coco_loader( data_dir, data_name, split=split, max_tokens=15)
    print('[DEBUG] Loading %s data ... %f secs' % (split, time.time() - t_start))
    
    data_loader = DataLoader(dataset=data, num_workers=nthreads,\
      batch_size=batchsize, shuffle=False, drop_last=True)
    
    #batchsize_cap = batchsize*ncap_per_img
    batchsize = batchsize
    max_tokens = data.max_tokens
    num_batches = np.int_(np.floor((len(data.ids)*1.)/batchsize))
    print('[DEBUG] Running inference on %s with %d batches' % (split, num_batches))
    
    total_loss = 0
    for batch_idx, (imgs, captions, _, mask, _) in tqdm(enumerate(data_loader), total=num_batches):
            
        pdb.set_trace()
        
        # print(np.shape(captions))
        # print(wordclass)
        # print(mask)
        #pdb.set_trace()
        imgs = imgs.view(batchsize, 3, 224, 224)
        
        imgs_v = Variable(imgs).cuda()
        

        
        imgsfeats, imgsfc7 = model_imgcnn(imgs_v)
        imgsfeats, imgsfc7 = repeat_img_per_cap(imgsfeats, imgsfc7, ncap_per_img)
        _, _, feat_h, feat_w = imgsfeats.size()


        pdb.set_trace()
      


        # if(attention == True):
        #   #Cross-entropy loss and attention loss of Show, Attend and Tell
        #     loss = F.cross_entropy(wordact_t[maskids, ...], \
        #     wordclass_t[maskids, ...].contiguous().view(maskids.shape[0])) \
        #     + (torch.sum(torch.pow(1. - torch.sum(attn, 1), 2)))\
        #     /(batchsize*feat_h*feat_w)
        # else:
        #     loss = F.cross_entropy(wordact_t[maskids, ...], \
        #     wordclass_t[maskids, ...].contiguous().view(maskids.shape[0]))

        #loss_train = loss_train + loss.data[0]

        # total_loss = total_loss + loss.data
        # print(loss.data)
        
    # true_captions, txt_true_captions = call_caption_text(data_dir, split)
    # txt_true_captions = txt_true_captions[0:len(list_pred_captions)]
    # BLEU_score = corpus_bleu( txt_true_captions , list_pred_captions)
    # BLEU_score,
        
    return  total_loss




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



