import torch
import os

os.environ["PATH"] = '/home/sepehr/texlive/2020/bin/x86_64-linux:' + os.environ["PATH"]  # DELETE IF NOT SEPEHR

import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image
import seaborn as sns

sns.set(style='whitegrid')
use_latex = True
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif"
}
if use_latex:
    plt.rcParams.update(tex_fonts)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = 'show_tell/checkpoints/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
word_map_file = 'data/saved_output/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
resize_trans = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = Image.open(image_path).convert("RGB")
    img = resize_trans(img)
    img = torch.FloatTensor(img).to(device)
    image = normalize(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = (top_k_words / vocab_size).long()  # (s)
        next_word_inds = (top_k_words % vocab_size).long()  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    complete_seqs = np.array(complete_seqs)

    return seq, alphas, np.flip(complete_seqs[np.argsort(complete_seqs_scores)])


def visualize_att(image_path, seq, alphas, rev_word_map, file_name, smooth=False):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([1024, 1024], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    fig = plt.figure(figsize=(13, 9))
    for t, word in enumerate(words):
        if t > 50:
            break
        ax = fig.add_subplot(np.ceil(len(words) / 12.), 12, t + 1)
        # plt.subplot(1, len(words), t + 1)
        if use_latex and word == '<start>':
            word = r'$<$start$>$'
        if use_latex and word == '<end>':
            word = r'$<$end$>$'
        ax.text(0, 1, word, color='black', backgroundcolor='white', fontsize=12)
        ax.imshow(image, aspect="auto")
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [1024, 1024])
        if t == 0:
            ax.imshow(alpha, alpha=0)
        else:
            ax.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{file_name}_attend.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.show()


def print_each_seq(complete_seqs, rev_word_map):
    for i, seq in enumerate(complete_seqs):
        words = [rev_word_map[ind] for ind in seq]
        sentence = " ".join(words[1:-1])
        print(f"Caption {i}: {sentence}")


if __name__ == '__main__':
    img_path = "show_tell/img/van.jpg"
    beam_size = 5

    # Load model
    checkpoint = torch.load(checkpoint)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # index to word

    # Encode, decode with attention and beam search
    seq, alphas, complete_seqs = caption_image_beam_search(encoder, decoder, img_path, word_map, beam_size)
    alphas = torch.FloatTensor(alphas)

    print_each_seq(complete_seqs, rev_word_map)
    # Visualize caption and attention of best sequence
    visualize_att(img_path, seq, alphas, rev_word_map, "van")
