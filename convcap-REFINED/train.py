import os
import os.path as osp
import numpy as np 
import time
 

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
                                                             

from coco_loader import coco_loader
from convcap import convcap, getVgg16
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter

#TODO
from val_test import test
import pdb



def repeat_img_per_cap(imgs_features, imgs_fc7, ncap_per_img):

    batchsize, featdim, feat_h, feat_w = imgs_features.size()
    batchsize_cap = batchsize*ncap_per_img
    imgs_features = imgs_features.unsqueeze(1).expand(\
      batchsize, ncap_per_img, featdim, feat_h, feat_w)
    imgs_features = imgs_features.contiguous().view(\
      batchsize_cap, featdim, feat_h, feat_w)
    
    batchsize, featdim = imgs_fc7.size()
    batchsize_cap = batchsize*ncap_per_img
    imgs_fc7 = imgs_fc7.unsqueeze(1).expand(\
      batchsize, ncap_per_img, featdim)
    imgs_fc7 = imgs_fc7.contiguous().view(\
      batchsize_cap, featdim)
  
    return imgs_features, imgs_fc7




def train(model_dir, gpu, data_folder, is_train, epochs, batchsize, ncap_per_img, num_layers, nthreads, finetune_after, learning_rate, lr_step_size, score_select, beam_size, attention):

    log_version = "default"    
    log_directory = 'logs'

    
    data_name = "coco_5_cap_per_img_5_min_word_freq"
     
    # DATA LOADING
    t_start = time.time()
    train_data = coco_loader(data_folder, data_name, split='TRAIN', max_tokens=15)
    print('[DEBUG] Loading train data ... %f secs' % (time.time() - t_start))
    
    train_data_loader = DataLoader(dataset=train_data, num_workers= nthreads, batch_size= batchsize, shuffle=True, drop_last=True)
    
    
    train_writer = SummaryWriter(os.path.join(log_directory, f"{log_version}/train"))
    val_writer = SummaryWriter(os.path.join(log_directory, f"{log_version}/val"))  

    #Load pre-trained VGG16 model
    model_imgcnn = getVgg16()  
    model_imgcnn.cuda() 
    model_imgcnn.train(True) 

    #Convcap model
    model_convcap = convcap(train_data.numwords, num_layers, is_attention=attention)
    model_convcap.cuda()
    model_convcap.train(True)

    optimizer = optim.RMSprop(model_convcap.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=.1)
    img_optimizer = None

    batchsize = batchsize
    ncap_per_img = ncap_per_img
    batchsize_cap = batchsize*ncap_per_img
    max_tokens = train_data.max_tokens
    
    nbatches = np.int_(np.floor((len(train_data.ids)*1.)/batchsize)) 



    for epoch in range(epochs):
        print(epoch)
        loss_train = 0.
    
        if(epoch == finetune_after):
            print("finetune")
            img_optimizer = optim.RMSprop(model_imgcnn.parameters(), lr=1e-5)
            img_scheduler = lr_scheduler.StepLR(img_optimizer, step_size=lr_step_size, gamma=.1)
    
        scheduler.step()    
        if(img_optimizer):
            print("optim")
            img_scheduler.step()
    
        #One epoch of train
        for batch_idx, (imgs, captions, wordclass, mask, _) in tqdm(enumerate(train_data_loader), total=nbatches):
            
           
            #pdb.set_trace()
            
            # print(np.shape(captions))
            # print(wordclass)
            # print(mask)
            #pdb.set_trace()
            imgs = imgs.view(batchsize, 3, 224, 224)
            wordclass = wordclass.view(batchsize_cap, max_tokens)
            mask = mask.view(batchsize_cap, max_tokens)
            
            imgs_v = Variable(imgs).cuda()
            word_class_v = Variable(wordclass).cuda()
            
            optimizer.zero_grad()
            if(img_optimizer):
                img_optimizer.zero_grad() 
            
            imgsfeats, imgsfc7 = model_imgcnn(imgs_v)
            imgsfeats, imgsfc7 = repeat_img_per_cap(imgsfeats, imgsfc7, ncap_per_img)
            _, _, feat_h, feat_w = imgsfeats.size()
    
            word_activated, _ = model_convcap(imgsfeats, imgsfc7, word_class_v)
    
            word_activated = word_activated[:,:,:-1]
            word_class_v = word_class_v[:,1:]
            mask = mask[:,1:].contiguous()
    
            word_activated_t = word_activated.permute(0, 2, 1).contiguous().view(\
              batchsize_cap*(max_tokens-1), -1)
            wordclass_t = word_class_v.contiguous().view(\
              batchsize_cap*(max_tokens-1), 1)
          
            maskids = torch.nonzero(mask.view(-1)).numpy().reshape(-1)
    

            loss = F.cross_entropy(word_activated_t[maskids, ...], \
            wordclass_t[maskids, ...].contiguous().view(maskids.shape[0]))
    

            loss_train = loss_train + loss.data
            print(loss.data)
            loss.backward()
    
            optimizer.step()
            if(img_optimizer):
                img_optimizer.step()
                
            train_writer.add_scalar('Train Iteration loss',
                            loss.data,
                            epoch * len(train_data_loader) + batch_idx + 1)        
                
        loss_train = loss_train/nbatches
        train_writer.add_scalar('Epoch Cross Entropy Loss',
                loss_train,
                epoch + 1)
                
                
            ##validation set
            
        #if (np.mod(batch_idx, 50) == 0 or batch_idx == nbatches-1) and batch_idx != 0: 
        print("Validation Score and Loss")
        model_text = 'model_' + str(epoch) + '.pth'
        modelfn = osp.join(model_dir, model_text)
        
        if(img_optimizer):
            img_optimizer_dict = img_optimizer.state_dict()
        else:
            img_optimizer_dict = None
        
        torch.save({
            'epoch': epoch,
            'state_dict': model_convcap.state_dict(),
            'img_state_dict': model_imgcnn.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'img_optimizer' : img_optimizer_dict,
          }, modelfn)
        BLEU_score, CE_loss = test(model_dir, data_folder, batchsize, 'VAL', num_layers, attention, nthreads=0, modelfn = modelfn, model_convcap=model_convcap, model_imgcnn=model_imgcnn)
        #CE_loss = test(model_dir, data_folder, batchsize, 'VAL', nthreads, attention, model_convcap=model_convcap, model_imgcnn=model_imgcnn)
        print("VALIDATION:", CE_loss)

        # Print status
        val_writer.add_scalar('BLEU-4 Score',
                            BLEU_score,
                            epoch + 1)
        val_writer.add_scalar('Validation Cross Entropy Loss',
                            CE_loss,
                            epoch + 1)
                

    BLEU_score, CE_loss = test(model_dir, data_folder, batchsize, 'TEST', num_layers, attention, nthreads=0, modelfn = modelfn, model_convcap=model_convcap, model_imgcnn=model_imgcnn)
    #CE_loss = test(model_dir, data_folder, batchsize, 'VAL', nthreads, attention, model_convcap=model_convcap, model_imgcnn=model_imgcnn)
    print("========= TEST SET RESULTS ===========")
    print("TEST:", BLEU_score)
    print("TEST:", CE_loss)

    # Print status
    # printing test results

            
    loss_train = (loss_train*1.)/(batch_idx)
    print('[DEBUG] Training epoch %d has loss %f' % (epoch, loss_train))

    modelfn = osp.join(model_dir, 'model.pth')

    if(img_optimizer):
        img_optimizer_dict = img_optimizer.state_dict()
    else:
        img_optimizer_dict = None

    torch.save({
        'epoch': epoch,
        'state_dict': model_convcap.state_dict(),
        'img_state_dict': model_imgcnn.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'img_optimizer' : img_optimizer_dict,
      }, modelfn)

    #Run on validation and obtain score
    # scores = test(args, 'val', model_convcap=model_convcap, model_imgcnn=model_imgcnn) 
    # score = scores[0][args.score_select]

    # if(score > bestscore):
    #   bestscore = score
    #   print('[DEBUG] Saving model at epoch %d with %s score of %f'\
    #     % (epoch, args.score_select, score))
    #   bestmodelfn = osp.join(args.model_dir, 'bestmodel.pth')
    #   os.system('cp %s %s' % (modelfn, bestmodelfn))


if __name__ == '__main__':

    model_dir = './models/'
    coco_root = './data/coco/'
    gpu = 0
    is_train = 1
    epochs = 30
    batchsize = 20
    ncap_per_img = 5
    num_layers =3
    nthreads = 0
    finetune_after = 8
    learning_rate = 5e-5
    lr_step_size = 15
    score_select = 'BLEU-4'
    beam_size = 1
    attention = False
    data_folder = "prep_data"
    
    train( model_dir, gpu, data_folder, is_train, epochs, batchsize, ncap_per_img, num_layers, nthreads, \
          finetune_after, learning_rate, lr_step_size, score_select, beam_size, attention)
    
    # train_data = coco_loader(coco_root, split='train', ncap_per_img=ncap_per_img)

    # train_data_loader = DataLoader(dataset=train_data, num_workers= nthreads,\
    #   batch_size= batchsize, shuffle=True, drop_last=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


