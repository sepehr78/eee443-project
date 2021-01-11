import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from sam import SAM
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.tensorboard import SummaryWriter

"""
Code for Show, Attend, and Tell, adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
"""

# Data parameters
data_folder = 'data/saved_output'
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint_path = 'show_tell/checkpoints'  # path where to save checkpoints
log_directory = 'show_tell/logs'

# Model parameters
use_glove = True  # whether to use pre-trained GloVe embedding in decoder
use_sam = True  # whether to use SAM optimizer
emb_dim = 300 if use_glove else 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 4  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = None if use_sam else 5.0  # do not clip when using SAM because direction of gradient is important in SAM
alpha_c = 1.0  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every xx batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = "show_tell/checkpoints/BEST_checkpoint_glove_coco_5_cap_per_img_5_min_word_freq.pth.tar"

# visualization params
log_name = "glove+sam"


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if use_sam:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout,
                                       use_glove=use_glove,
                                       word_map=word_map)
        base_optimizer = torch.optim.SGD
        decoder_optimizer = SAM(filter(lambda p: p.requires_grad, decoder.parameters()), base_optimizer,
                                lr=decoder_lr, momentum=0.9)

        checkpoint = torch.load(checkpoint)
        encoder = checkpoint['encoder']
        encoder_optimizer = None
        print("Loading best encoder but random decoder and using SAM...")

    elif checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout,
                                       use_glove=use_glove,
                                       word_map=word_map)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Continuing training from epoch {start_epoch}...")
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        if use_sam:
            lr = checkpoint['decoder_optimizer'].param_groups[0]['lr']
            base_optimizer = torch.optim.SGD
            decoder_optimizer = SAM(filter(lambda p: p.requires_grad, decoder.parameters()), base_optimizer,
                                    lr=lr, momentum=0.9)
        else:
            decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']

        if use_sam and fine_tune_encoder is True:
            lr = checkpoint['encoder_optimizer'].param_groups[0]['lr']
            base_optimizer = torch.optim.SGD
            encoder_optimizer = SAM(filter(lambda p: p.requires_grad, encoder.parameters()), base_optimizer,
                                    lr=lr, momentum=0.9)
        else:
            encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            if use_sam:
                base_optimizer = torch.optim.SGD
                encoder_optimizer = SAM(filter(lambda p: p.requires_grad, encoder.parameters()), base_optimizer,
                                        lr=encoder_lr, momentum=0.9)
            else:
                encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                     lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # initialize dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CocoCaptionDataset(data_folder, data_name, 'TRAIN', transforms=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CocoCaptionDataset(data_folder, data_name, 'VAL', transforms=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    print(f"Train dataloader len: {len(train_loader)}")
    print(f"Val dataloader len: {len(val_loader)}")

    # set up tensorbaord
    train_writer = SummaryWriter(os.path.join(log_directory, f"{log_name}/train"))
    val_writer = SummaryWriter(os.path.join(log_directory, f"{log_name}/val"))

    # Epochs
    for epoch in tqdm(range(start_epoch, epochs)):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              train_writer=train_writer)

        # One epoch's validation
        recent_bleu4, val_loss, val_top5_acc = validate(val_loader=val_loader,
                                                        encoder=encoder,
                                                        decoder=decoder,
                                                        criterion=criterion)
        val_writer.add_scalar('Epoch loss', val_loss, epoch + 1)
        val_writer.add_scalar('Epoch top-5 accuracy', val_top5_acc, epoch + 1)
        val_writer.add_scalar('BLEU-4', recent_bleu4, epoch + 1)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        checkpoint_name = data_name
        if use_glove:
            checkpoint_name = f"glove_{checkpoint_name}"
        if use_sam:
            checkpoint_name = f"sam_{checkpoint_name}"
        save_checkpoint(checkpoint_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best, checkpoint_path)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, train_writer):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    :param train_writer: tensorboard SummaryWriter
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (og_imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        og_imgs = og_imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(og_imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        if use_sam:
            decoder_optimizer.first_step(zero_grad=True)
            if encoder_optimizer is not None:
                encoder_optimizer.first_step(zero_grad=True)

            # perform second pass needed for SAM
            imgs = encoder(og_imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            second_loss = criterion(scores, targets)
            second_loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            second_loss.backward()

            # perform second update
            decoder_optimizer.second_step(zero_grad=True)
            if encoder_optimizer is not None:
                encoder_optimizer.second_step(zero_grad=True)
        else:
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        # Print status
        train_writer.add_scalar('Iteration loss',
                                losses.val,
                                epoch * len(train_loader) + i + 1)
        train_writer.add_scalar('Iteration top-5 accuracy',
                                top5accs.val,
                                epoch * len(train_loader) + i + 1)

        train_writer.add_scalar('Batch processing duration',
                                batch_time.val,
                                epoch * len(train_loader) + i + 1)
        train_writer.add_scalar('Data loading duration',
                                data_time.val,
                                epoch * len(train_loader) + i + 1)
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
        start = time.time()

    train_writer.add_scalar('Epoch loss', losses.avg, epoch + 1)
    train_writer.add_scalar('Epoch top-5 accuracy', top5accs.avg, epoch + 1)


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data  # TODO PRAY THIS IS CORRECT

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))
            start = time.time()

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)
    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))

    return bleu4, losses.avg, top5accs.avg


if __name__ == '__main__':
    main()
